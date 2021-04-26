import time
import numpy as np
import torch

a = torch.randn((64, 200, 3))
b = torch.randn((64, 3))

layer = torch.nn.Linear(3, 64)

times = []
for i in range(100):
    start = time.perf_counter()
    _ = layer(a)
    end = time.perf_counter()
    times.append(end - start)
print(np.mean(times) * 200)

times = []
for i in range(100):
    start = time.perf_counter()
    _ = layer(b)
    end = time.perf_counter()
    times.append(end - start)
print(np.mean(times) * 200)