import mlx.nn as nn
import mlx.core as mx
import sys

# Refined Monkey-patch
if not hasattr(nn.QuantizedLinear, "quantize_module"):
    print("Fixing missing quantize_module with predicate...")
    def quantize_module(model, **kwargs):
        # Only quantize Linear layers to avoid breaking Embeddings
        return nn.quantize(
            model, 
            **kwargs, 
            class_predicate=lambda p, m: isinstance(m, nn.Linear)
        )
    nn.QuantizedLinear.quantize_module = staticmethod(quantize_module)

try:
    from lightning_whisper_mlx import LightningWhisperMLX
    print("Initializing model...")
    # This should now succeed and handle the weights correctly
    model = LightningWhisperMLX(model="large-v3", batch_size=12, quant="4bit")
    print("Initialization success!")
    
    # Try a fake transcription or just check the embedding layer type
    print(f"Embedding type: {type(model.model.decoder.token_embedding)}")
    if isinstance(model.model.decoder.token_embedding, nn.Embedding):
        print("Success: Embedding remains unquantized (float16).")
    else:
        print(f"Failure: Embedding is {type(model.model.decoder.token_embedding)}")
        
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
