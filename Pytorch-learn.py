import torch
start_tensor = torch.rand(2,5)
new_tensor = start_tensor.fill_(5)
new_tensor.sqrt_()
x = torch.linspace(0.1, 10, 15, )
tensor_chunk = torch.chunk(x, 4, 0)
print(tensor_chunk)
