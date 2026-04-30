import time
import queue
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import os

class RT:
    def __init__(self):
        self.q = queue.Queue()
        self.recording = False
        
    def cb(self, indata, frames, time_info, status):
        if status:
            print("Status:", status)
        self.q.put(indata.copy())
        
    def start(self):
        self.recording = True
        self.stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=self.cb)
        self.stream.start()
        self.t = threading.Thread(target=self.loop)
        self.t.start()
        
    def stop(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.t.join()
        
    def loop(self):
        from lightning_whisper_mlx import LightningWhisperMLX
        print("Loading model...")
        model = LightningWhisperMLX(model="large-v3", batch_size=12, quant="4bit")
        print("Model loaded.")
        
        audio_buffer = np.array([], dtype=np.float32)
        committed_text = ""
        
        last_process_time = time.time()
        
        while self.recording or not self.q.empty():
            while not self.q.empty():
                audio_buffer = np.concatenate((audio_buffer, self.q.get().flatten()))
                
            now = time.time()
            if len(audio_buffer) >= 16000 * 2 and (now - last_process_time > 2.0):
                # We have at least 2 seconds of audio and 2 seconds passed
                # Transcribe
                temp_wav = "test_rt.wav"
                wav.write(temp_wav, 16000, (audio_buffer * 32767).astype(np.int16))
                res = model.transcribe(temp_wav)
                text = res.get("text", "").strip()
                print("---")
                print(committed_text + " " + text)
                
                # If buffer gets too long, commit it
                if len(audio_buffer) > 16000 * 15: # 15 seconds
                    committed_text += " " + text
                    audio_buffer = np.array([], dtype=np.float32)
                    
                last_process_time = now
            else:
                time.sleep(0.1)

if __name__ == "__main__":
    rt = RT()
    rt.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rt.stop()
        print("Stopped")
