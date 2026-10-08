"""Spike S3/S4: do the custom CUDA ops compile, and does grid_sample have a double backward?"""
import time

import torch
import torch.nn.functional as F

from torch_utils.ops import bias_act, upfirdn2d

dev = torch.device("cuda")

# S3: bias_act and upfirdn2d (impl="cuda" falls back to "ref" with a warning if the build fails)
x = torch.randn(8, 64, 128, 128, device=dev, requires_grad=True)
b = torch.randn(64, device=dev, requires_grad=True)
print("bias_act plugin:", bias_act._init())
print("upfirdn2d plugin:", upfirdn2d._init())
f = upfirdn2d.setup_filter([1, 3, 3, 1], device=dev)
for impl in ("ref", "cuda"):
    for _ in range(3):  # warm-up
        y = upfirdn2d.upsample2d(bias_act.bias_act(x, b, act="lrelu", impl=impl), f, impl=impl)
        y.square().sum().backward()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        y = upfirdn2d.upsample2d(bias_act.bias_act(x, b, act="lrelu", impl=impl), f, impl=impl)
        y.square().sum().backward()
    torch.cuda.synchronize()
    print(f"{impl}: {(time.time() - t) / 20 * 1e3:.2f} ms/iter")
yr = upfirdn2d.upsample2d(bias_act.bias_act(x, b, act="lrelu", impl="ref"), f, impl="ref")
yc = upfirdn2d.upsample2d(bias_act.bias_act(x, b, act="lrelu", impl="cuda"), f, impl="cuda")
print("max |cuda - ref|:", (yr - yc).abs().max().item())

# S4: double backward through grid_sample (ADA geometric augment + R1)
x = torch.randn(4, 3, 64, 64, device=dev, requires_grad=True)
theta = torch.eye(2, 3, device=dev).unsqueeze(0).repeat(4, 1, 1) + 0.05 * torch.randn(4, 2, 3, device=dev)
grid = F.affine_grid(theta, list(x.shape), align_corners=False)
try:
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    (g,) = torch.autograd.grad(y.square().sum(), x, create_graph=True)
    g.square().sum().backward()
    print("grid_sample double backward: OK, |x.grad| =", x.grad.norm().item())
except Exception as e:
    print("grid_sample double backward: FAILED:", type(e).__name__, e)
