import mlx.nn as nn
import sys

# Monkey-patch
if not hasattr(nn.QuantizedLinear, "quantize_module"):
    print("Fixing missing quantize_module...")
    nn.QuantizedLinear.quantize_module = staticmethod(nn.quantize)

try:
    from lightning_whisper_mlx import LightningWhisperMLX
    print("Initializing model...")
    model = LightningWhisperMLX(model="large-v3", batch_size=12, quant="4bit")
    print("Initialization success!")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
