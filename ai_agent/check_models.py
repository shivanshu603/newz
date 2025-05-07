from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def list_local_models():
    """List all locally cached Hugging Face models"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
    
    if not os.path.exists(cache_dir):
        print("No models found in Hugging Face cache")
        return []
    
    models = []
    for root, dirs, files in os.walk(cache_dir):
        if "config.json" in files:
            model_path = os.path.relpath(root, cache_dir)
            models.append(model_path)
    
    return models

if __name__ == "__main__":
    print("Checking for locally installed models...")
    models = list_local_models()
    
    if models:
        print("\nFound these models:")
        for model in models:
            print(f"- {model}")
    else:
        print("\nNo pre-downloaded models found.")
        
    print("\nAvailable CUDA device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Available RAM:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB" if torch.cuda.is_available() else "N/A")
