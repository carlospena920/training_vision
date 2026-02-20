import torch

print("CUDA disponible:", torch.cuda.is_available())
print("GPU detectada:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Ninguna")
print("Dispositivo actual:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
