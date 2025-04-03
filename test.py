import torch
print(torch.cuda.is_available())  # Sollte `True` zurückgeben
print(torch.cuda.device_count())  # Sollte >= 1 sein
print(torch.cuda.get_device_name(0))  # Sollte deine AMD GPU anzeigen