import inspect
from transformers import AutoModel

HF_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"

def inspect_model_source():
    model = AutoModel.from_pretrained(HF_MODEL_NAME)
    print("Model type:", type(model))
    
    print("\n--- Vjepa2Model Forward Method ---")
    try:
        print(inspect.getsource(model.forward))
    except Exception as e:
        print("Could not get model.forward source:", e)

    print("\n--- Vjepa2Encoder/VisionTransformer Info ---")
    # Let's find where the blocks are
    for name, child in model.named_children():
        print(f"Child name: {name}, type: {type(child)}")
        
    if hasattr(model, "encoder"):
        print("\n--- encoder.layer[0] Info ---")
        layer0 = model.encoder.layer[0]
        print("layer0 type:", type(layer0))
        try:
            print("layer0 forward source:")
            print(inspect.getsource(layer0.forward))
        except Exception as e:
            print("Could not get layer0.forward source:", e)
            
        if hasattr(layer0, "attention"):
            print("\n--- layer0.attention Info ---")
            attn = layer0.attention
            print("attention type:", type(attn))
            try:
                print("attention forward source:")
                print(inspect.getsource(attn.forward))
            except Exception as e:
                print("Could not get attention.forward source:", e)

if __name__ == "__main__":
    inspect_model_source()
