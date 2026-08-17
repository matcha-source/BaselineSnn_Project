import torch

print("PyTorch version:")
print(torch.__version__)

print("CUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))