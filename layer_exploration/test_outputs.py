import os
import torch
from transformers import AutoModel, AutoVideoProcessor

HF_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"

def test_model_outputs():
    print("Loading model and processor...")
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    model = AutoModel.from_pretrained(HF_MODEL_NAME, attn_implementation="eager")
    model.eval()
    
    # Create a dummy video tensor: Batch x Time x Channels x Height x Width
    # Standard shape: 1 x 8 x 3 x 224 x 224
    dummy_video = torch.randn(1, 8, 3, 224, 224)
    
    # Process inputs
    # Note: processor expects frames as [T, C, H, W] for a single video, or list of frames
    inputs = processor(list(dummy_video[0]), return_tensors="pt")
    
    print("\nModel configuration:")
    for k, v in model.config.to_dict().items():
        if k in ["num_attention_heads", "num_hidden_layers", "hidden_size", "tubelet_size", "patch_size"]:
            print(f"  {k}: {v}")
            
    # Register hooks to capture attentions
    attention_maps = {}
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (context_layer, attention_probs)
            # attention_probs shape is: [batch_size, num_heads, num_tokens, num_tokens]
            attention_maps[layer_idx] = output[1].detach().cpu()
        return hook

    for idx, layer in enumerate(model.encoder.layer):
        layer.attention.register_forward_hook(make_hook(idx))

    print("\nRunning forward pass with forward hooks registered...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    print("\nOutputs keys:")
    print("  ", list(outputs.keys()))
    
    if "hidden_states" in outputs:
        print(f"  hidden_states length: {len(outputs.hidden_states)}")
        print(f"  hidden_states[0] shape: {outputs.hidden_states[0].shape}")
        
    print(f"\nCaptured attentions via hooks: {len(attention_maps)} layers")
    if len(attention_maps) > 0:
        print(f"  Layer 0 attention shape: {attention_maps[0].shape}")


if __name__ == "__main__":
    test_model_outputs()
