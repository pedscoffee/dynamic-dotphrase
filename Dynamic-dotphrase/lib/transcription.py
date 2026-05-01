"""
Whisper transcription — Apple Silicon optimised via lightning-whisper-mlx.
Supports both file-based and real-time streaming transcription.
"""

import gc
import os
import time
import queue
import threading
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import streamlit as st

AUDIO_TEMP_FILE = Path(__file__).parent.parent / ".tmp_recording.wav"


def save_audio_widget_output(audio_bytes_io) -> Path:
    """Write the BytesIO from st.audio_input() directly to a WAV file."""
    audio_bytes_io.seek(0)
    AUDIO_TEMP_FILE.write_bytes(audio_bytes_io.read())
    return AUDIO_TEMP_FILE


class RealtimeTranscriber:
    """Singleton for managing background audio recording and realtime transcription."""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.thread = None
        self.model = None
        
        self.committed_text = ""
        self.current_text = ""
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("[Audio Status]", status)
        self.audio_queue.put(indata.copy())
        
    def start(self):
        """Starts the background recording and transcription loop."""
        if self.is_recording:
            return
        self.is_recording = True
        self.committed_text = ""
        self.current_text = ""
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Clear queue
        while not self.audio_queue.empty():
            self.audio_queue.get()
            
        try:
            self.stream = sd.InputStream(
                samplerate=16000, 
                channels=1, 
                dtype='float32', 
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            st.error(f"Failed to start audio stream: {e}")
            self.is_recording = False
            return
            
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stops the recording."""
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.thread:
            self.thread.join(timeout=5.0)
            self.thread = None
            
    def get_transcript(self) -> str:
        """Returns the full transcript constructed so far."""
        return (self.committed_text + " " + self.current_text).strip()
        
    def unload_model(self):
        """Frees the MLX model from RAM."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            
    def _patch_mlx(self):
        """Patch for MLX backward compatibility."""
        try:
            import mlx.nn as nn
            if not hasattr(nn.QuantizedLinear, "quantize_module"):
                def quantize_module(model, **kwargs):
                    return nn.quantize(
                        model, **kwargs,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear),
                    )
                nn.QuantizedLinear.quantize_module = staticmethod(quantize_module)
        except ImportError:
            pass

    def _process_loop(self):
        """Background thread that transcribes accumulated audio every few seconds."""
        # Ensure ffmpeg is in PATH
        paths = ["/opt/homebrew/bin", "/usr/local/bin"]
        current_path = os.environ.get("PATH", "")
        for p in paths:
            if p not in current_path:
                current_path = f"{p}:{current_path}"
        os.environ["PATH"] = current_path
        
        self._patch_mlx()
        
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
        except ImportError:
            print("lightning-whisper-mlx not installed")
            return
            
        if self.model is None:
            print("[DEBUG] Loading Whisper for realtime (large-v3, 4-bit)...")
            self.model = LightningWhisperMLX(model="large-v3", batch_size=12, quant="4bit")
            
        last_process_time = time.time()
        temp_wav = str(AUDIO_TEMP_FILE)
        
        try:
            while self.is_recording or not self.audio_queue.empty():
                # Drain queue into buffer
                while not self.audio_queue.empty():
                    self.audio_buffer = np.concatenate((self.audio_buffer, self.audio_queue.get().flatten()))
                    
                now = time.time()
                # Process if we have >= 2 seconds of audio and 2 seconds have passed
                if len(self.audio_buffer) >= 16000 * 2 and (now - last_process_time > 2.0):
                    try:
                        wav.write(temp_wav, 16000, (self.audio_buffer * 32767).astype(np.int16))
                        res = self.model.transcribe(temp_wav)
                        self.current_text = res.get("text", "").strip()
                    except Exception as e:
                        print(f"[DEBUG] Transcription error: {e}")
                        
                    # Commit text every 30 seconds to prevent buffer from growing forever
                    if len(self.audio_buffer) >= 16000 * 30:
                        self.committed_text += " " + self.current_text
                        self.current_text = ""
                        self.audio_buffer = np.array([], dtype=np.float32)
                        
                    last_process_time = now
                else:
                    time.sleep(0.1)
                    
            # Final process on stop
            if len(self.audio_buffer) > 0:
                try:
                    wav.write(temp_wav, 16000, (self.audio_buffer * 32767).astype(np.int16))
                    res = self.model.transcribe(temp_wav)
                    self.current_text = res.get("text", "").strip()
                except Exception:
                    pass
                self.committed_text += " " + self.current_text
                self.current_text = ""
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
            print("[DEBUG] Realtime transcription stopped, temporary audio destroyed.")


def transcribe_audio(wav_path: Path) -> str:
    """
    Standard one-shot transcription for uploaded/finished audio files.
    """
    try:
        paths = ["/opt/homebrew/bin", "/usr/local/bin"]
        current_path = os.environ.get("PATH", "")
        for p in paths:
            if p not in current_path:
                current_path = f"{p}:{current_path}"
        os.environ["PATH"] = current_path

        try:
            import mlx.nn as nn
            if not hasattr(nn.QuantizedLinear, "quantize_module"):
                def quantize_module(model, **kwargs):
                    return nn.quantize(
                        model, **kwargs,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear),
                    )
                nn.QuantizedLinear.quantize_module = staticmethod(quantize_module)
        except ImportError:
            pass

        from lightning_whisper_mlx import LightningWhisperMLX
    except ImportError:
        st.error(
            "`lightning-whisper-mlx` is not installed. "
            "Run: `pip install lightning-whisper-mlx`"
        )
        return ""

    # Ensure background model is unloaded if standard transcribe is called
    rt = RealtimeTranscriber.get_instance()
    rt.unload_model()

    with st.spinner("Loading Whisper model (large-v3) [4-bit]…"):
        model = LightningWhisperMLX(model="large-v3", batch_size=12, quant="4bit")

    with st.spinner("Transcribing audio…"):
        print(f"[DEBUG] Starting transcription of {wav_path}")
        result = model.transcribe(str(wav_path))
        transcript = result.get("text", "").strip()
        print(f"[DEBUG] Transcription finished. Length: {len(transcript)} chars")

    del model
    gc.collect()

    return transcript
