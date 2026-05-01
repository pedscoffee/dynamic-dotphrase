from lightning_whisper_mlx import LightningWhisperMLX
import sys

try:
    print("Testing initialization of large-v3 with 4bit quantization...")
    # We don't need to actually transcribe, just see if __init__ works.
    # Note: This might try to download the model if not present.
    model = LightningWhisperMLX(model="large-v3", batch_size=12, quant="4bit")
    print("Successfully initialized large-v3 [4bit]!")
    sys.exit(0)
except Exception as e:
    print(f"Error initializing model: {e}")
    sys.exit(1)
