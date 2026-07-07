import inspect
from transformers import AutoModel

HF_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"

def inspect_embeddings():
    model = AutoModel.from_pretrained(HF_MODEL_NAME)
    embeddings = model.encoder.embeddings.patch_embeddings
    print("Embeddings type:", type(embeddings))
    try:
        print("\nEmbeddings forward source:")
        print(inspect.getsource(embeddings.forward))
    except Exception as e:
        print("Could not get embeddings.forward source:", e)

if __name__ == "__main__":
    inspect_embeddings()
