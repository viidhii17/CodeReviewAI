import torch
import transformers
import datasets
import fastapi
import tree_sitter
import gradio

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("All imports OK!")