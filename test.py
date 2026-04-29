import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpu_branch = nn.Linear(512, 256).cuda()
        self.cpu_branch = nn.Linear(512, 256)  # stays on CPU
        self.fusion = nn.Linear(512, 10).cuda()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def forward(self, x_gpu):

        gpu_device = x_gpu.device

        x_cpu = x_gpu.detach().cpu()  # move input to CPU (async-friendly)

        # Launch CPU work in a background thread
        cpu_future = self.executor.submit(self.cpu_branch, x_cpu)

        # GPU continues immediately, no blocking
        gpu_out = self.gpu_branch(x_gpu)

        # Sync point: retrieve CPU result and move to GPU
        cpu_out = cpu_future.result().to(gpu_device)

        # Combine and continue on GPU
        combined = torch.cat([gpu_out, cpu_out], dim=-1)
        return self.fusion(combined)
    
device = "cuda"

model = HybridModel()

batch_size = 10
input_dim = 512

input = torch.rand(size=(batch_size, input_dim))

output = model(input.to(device))

print("Done!")