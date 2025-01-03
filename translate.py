import pyaudio
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import queue
import logging
import time
import os
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 设置全局代理
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:3128'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:3128'
# http://username:password@your_proxy_address:your_proxy_port
# os.environ['HTTP_PROXY'] = 'rb-proxy-szh.bosch.com:8080'
# os.environ['HTTPS_PROXY'] = 'rb-proxy-szh.bosch.com:8080'

# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''

class ConfigManager:
    """配置管理类"""
    @staticmethod
    def init_api_key():
        """从ApiKey.conf文件初始化API Key"""
        try:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建配置文件路径
            config_path = os.path.join(current_dir, 'ApiKey.conf')
            
            # 检查文件是否存在
            if not os.path.exists(config_path):
                logger.error("ApiKey.conf文件未找到")
                return False
                
            # 读取文件内容
            with open(config_path, 'r') as f:
                api_key = f.read().strip()
                
            # 验证API Key格式
            if not api_key or len(api_key) < 20:  # 简单长度验证
                logger.error("无效的API Key格式")
                return False
                
            # 设置API Key
            dashscope.api_key = api_key
            logger.debug("API Key初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化API Key失败: {str(e)}")
            return False

class AudioProcessor:
    def __init__(self):
        self.recognition = None
        self.is_running = False
        self.audio_source = 'system'
        self.volume_threshold = 0.02
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.buffer_size = 4096
        self.input_device = None
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.audio_thread = None
        self.send_thread = None
        self.shutdown_event = threading.Event()
        self.audio_callback = self._audio_callback 

    def get_audio_devices(self):
        """获取可用的音频输入设备"""
        devices = []
        mic_device = None
        mix_device = None
        default_device = None
        
        # 获取默认输入设备
        default_input = self.pyaudio.get_default_input_device_info()
        
        for i in range(self.pyaudio.get_device_count()):
            device_info = self.pyaudio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                device_name = device_info['name'].lower()
                
                # 检查是否是默认设备
                if device_info['name'] == default_input['name']:
                    default_device = (i, device_info['name'])
                
                # 检查是否是麦克风设备（考虑中英文）
                if not mic_device and ('mic' in device_name or '麦克风' in device_name):
                    mic_device = (i, device_info['name'])
                
                # 检查是否是混音设备（考虑中英文）
                if not mix_device and ('mix' in device_name or '混音' in device_name):
                    mix_device = (i, device_info['name'])
                
                # 如果所有设备都找到了就退出循环
                if default_device and mic_device and mix_device:
                    break
        
        # 添加找到的设备，优先使用默认设备
        if default_device:
            devices.append(default_device)
        if mic_device and mic_device != default_device:
            devices.append(mic_device)
        if mix_device and mix_device != default_device:
            devices.append(mix_device)
            
        return devices

    def set_input_device(self, device_id):
        """设置输入设备"""
        try:
            device_info = self.pyaudio.get_device_info_by_index(device_id)
            if device_info['maxInputChannels'] == 0:
                raise ValueError("Selected device is not an input device")
            
            # 测试设备是否可用
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.buffer_size,
                start=False
            )
            stream.stop_stream()
            stream.close()
            
            self.input_device = device_id
            logger.debug(f"Input device set to: {device_id} ({device_info['name']})")
            return True
        except Exception as e:
            logger.error(f"Invalid device {device_id}: {str(e)}")
            self.input_device = None
            return False

    def start_recognition(self, callback):
        logger.debug("Starting recognition...")
        self.callback = callback
        try:
            if self.input_device is None:
                raise ValueError("No valid input device selected")
            
            self.recognition = Recognition(
                model='paraformer-realtime-v2',
                format='pcm',
                sample_rate=self.sample_rate,
                bit_width=16,
                callback=RecognitionHandler(callback)
            )
            self.recognition.start()
            self.is_running = True
            self.shutdown_event.clear()
            
            # 启动音频捕获线程
            self.audio_thread = threading.Thread(target=self.audio_capture)
            self.audio_thread.start()
            
            # 启动音频发送线程
            self.send_thread = threading.Thread(target=self.send_audio)
            self.send_thread.start()
            
            logger.debug("Recognition started successfully")
        except Exception as e:
            logger.error(f"Recognition start failed: {str(e)}")
            self.stop_recognition()

    def stop_recognition(self):
        logger.debug("Stopping recognition...")
        self.is_running = False
        self.shutdown_event.set()
        
        # 停止识别
        if self.recognition:
            try:
                self.recognition.stop()
            except Exception as e:
                logger.error(f"Error while stopping recognition: {str(e)}")
            finally:
                self.recognition = None
        
        # 停止音频流
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {str(e)}")
            finally:
                self.stream = None
        
        # 等待线程结束
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.5)
            if self.audio_thread.is_alive():
                logger.warning("Audio thread did not stop in time")
        
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=0.5)
            if self.send_thread.is_alive():
                logger.warning("Send thread did not stop in time")
        
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.debug("Recognition stopped")

    def audio_capture(self):
        logger.debug("Starting audio capture...")
        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.buffer_size,
                stream_callback=self.audio_callback,
                start=False
            )
            self.stream.start_stream()
            while self.is_running and not self.shutdown_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio capture error: {str(e)}")
        finally:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logger.error(f"Error closing audio stream: {str(e)}")
                finally:
                    self.stream = None
            logger.debug("Audio capture stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        try:
            if self.is_running:
                # 将音频数据放入队列
                self.audio_queue.put(in_data)
                # 计算音量
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                volume = np.abs(audio_data).mean() / 32768
                self.callback.update_volume(volume)
            return (None, pyaudio.paContinue)
        except Exception as e:
            logger.error(f"Audio callback error: {str(e)}")
            return (None, pyaudio.paAbort)

    def send_audio(self):
        logger.debug("Starting audio sending...")
        while self.is_running or not self.audio_queue.empty():
            if self.shutdown_event.is_set():
                break
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
                if audio_data and self.recognition:
                    self.recognition.send_audio_frame(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio sending error: {str(e)}")
        logger.debug("Audio sending stopped")

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("实时语音识别")
        self.geometry("800x600")
        
        self.audio_processor = AudioProcessor()
        
        # 创建UI
        self.create_widgets()
        
        # 绑定窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 设备选择
        ttk.Label(self, text="选择输入设备:").pack()
        self.device_var = tk.StringVar()
        self.device_combobox = ttk.Combobox(
            self, 
            textvariable=self.device_var,
            width=60,
            state='readonly'
        )
        self.device_combobox.pack(pady=5)
        
        # 刷新设备列表
        self.refresh_devices()
        
        # 音量显示
        self.volume_canvas = tk.Canvas(self, width=200, height=20, bg='white')
        self.volume_canvas.pack()
        
        # 主文本框
        self.text_box = tk.Text(self)
        self.text_box.pack(expand=True, fill='both')
        
        # 未完成句子显示框
        self.pending_text = tk.Text(self, height=6, bg='#f0f0f0')
        self.pending_text.pack(fill='x')
        
        # 控制按钮
        self.start_btn = ttk.Button(self, text="开始", command=self.start)
        self.start_btn.pack(side='left')
        
        self.stop_btn = ttk.Button(self, text="停止", command=self.stop)
        self.stop_btn.pack(side='right')
    
    def refresh_devices(self):
        """刷新音频设备列表"""
        devices = self.audio_processor.get_audio_devices()
        self.device_combobox['values'] = [f"{device[0]}: {device[1]}" for device in devices]
        if devices:
            self.device_combobox.current(0)
    
    def start(self):
        # 获取选择的设备ID
        device_str = self.device_var.get()
        if device_str:
            device_id = int(device_str.split(':')[0])
            if not self.audio_processor.set_input_device(device_id):
                self.text_box.insert('end', "错误：无法使用该音频设备，请选择其他设备\n")
                self.text_box.see('end')
                return
            self.audio_processor.start_recognition(self)
        
    def stop(self):
        self.audio_processor.stop_recognition()
        
    def update_text(self, text):
        self.text_box.insert('end', text + '\n')
        self.text_box.see('end')
        
    def update_volume(self, volume):
        self.volume_canvas.delete('all')
        width = min(int(volume * 100), 200)
        self.volume_canvas.create_rectangle(0, 0, width, 20, fill='green')

    def update_pending_text(self, text):
        """更新未完成句子显示框"""
        self.pending_text.delete('1.0', 'end')
        self.pending_text.insert('end', text)
        self.pending_text.see('end')

    def on_closing(self):
        """窗口关闭事件处理"""
        logger.debug("Closing application...")
        self.audio_processor.stop_recognition()
        try:
            self.destroy()
        except Exception as e:
            logger.error(f"Error closing application: {str(e)}")
        finally:
            os._exit(0)

class RecognitionHandler(dashscope.audio.asr.RecognitionCallback):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.current_sentence = ""  # 当前正在识别的句子
        self.translation_client = OpenAI(
            api_key=dashscope.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def translate_text(self, text):
        """调用翻译API将文本翻译成中文"""
        try:
            messages = [{"role": "user", "content": text}]
            translation_options = {
                "source_lang": "English",  # 假设输入是英文
                "target_lang": "Chinese"
            }
            
            completion = self.translation_client.chat.completions.create(
                model="qwen-mt-turbo",
                messages=messages,
                extra_body={
                    "translation_options": translation_options
                }
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return None

    def on_event(self, result):
        try:
            sentence = result.get_sentence()
            if sentence:
                current_text = sentence['text']
                
                # 更新当前句子
                self.current_sentence = current_text
                
                # 使用result的is_sentence_end方法判断句子是否结束
                if result.is_sentence_end(sentence):
                    # 句子完成，添加到主文本框
                    self.callback.update_text("\n原文: " + self.current_sentence)
                    
                    # 调用翻译API
                    translated_text = self.translate_text(self.current_sentence)
                    if translated_text:
                        self.callback.update_text("译文: " + translated_text + "\n")
                    
                    # 清空未完成句子显示框
                    self.callback.update_pending_text("")
                else:
                    # 句子未完成，更新未完成句子显示框
                    self.callback.update_pending_text(self.current_sentence)
                
                logger.debug(f"Received sentence: {current_text}")
        except Exception as e:
            logger.error(f"Error processing recognition result: {str(e)}")

    def on_complete(self):
        logger.debug("Recognition completed")

    def on_error(self, error):
        logger.error(f"Recognition error: {str(error)}")

if __name__ == "__main__":
    if not ConfigManager.init_api_key():
        logger.error("无法初始化API Key，程序可能无法正常工作")
    else:
        app = Application()
        app.mainloop()