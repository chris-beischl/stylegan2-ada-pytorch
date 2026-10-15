"""PROBE: batched, seeded projection of images into W, derived from project() in projector.py.

Imported by PROBE from this checkout (a module in probe/ would clash with PROBE's package `probe`).

Differences to project():
- many targets in one optimisation; each target has its own w and its own loss term, so targets
  do not interact (results equal separate runs up to floating-point differences);
- class vectors c for a conditional G; the W statistics are computed per class;
- randomness only from `seed`, one stream per target row of the call (so `batch_size` does not
  change the draws);
- VGG16 (LPIPS features) from a local file (probe/fetch_vgg16.sh), no download at run time;
- G's noise inputs stay fixed (noise_mode='const') instead of being optimised: optimised noise is
  shared by the whole batch and lives in a copy of G, so the returned w would not reproduce their
  images with the unmodified G, which PLAN uses to render;
- init "mean" (as project(): class-mean w) or "sample" (w of a random z of the class): several
  projections of one target need different starting points;
- optional bf16 autocast for VGG16 (`vgg_bf16`, ~8 % faster at equal LPIPS distance, other
  numbers; probe/progress.md "Performance (2026-10)").
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F


def _generator(seed, key, device):
    state = np.random.SeedSequence(seed, spawn_key=key).generate_state(1, np.uint64)[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(int(state))
    return gen


def _features(vgg16, images, bf16=False):
    if images.shape[2] > 256:  # VGG was built for 224 x 224 images
        images = F.interpolate(images, size=(256, 256), mode='area')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        return vgg16(images, resize_images=False, return_lpips=True).float()


def project_batch(
    G,
    targets: torch.Tensor,          # uint8 [N, C, H, W] at G's resolution
    c,                              # [N, c_dim] class vectors, or None for an unconditional G
    *,
    seed: int,
    device: torch.device,
    vgg16_path: str,
    num_steps: int = 1000,
    init: str = 'mean',             # 'mean' | 'sample'
    batch_size: int = 16,
    vgg_bf16: bool = False,
    w_avg_samples: int = 10000,
    initial_learning_rate: float = 0.1,
    initial_noise_factor: float = 0.05,
    lr_rampdown_length: float = 0.25,
    lr_rampup_length: float = 0.05,
    noise_ramp_length: float = 0.75,
):
    """Return (w [N, w_dim], dist [N]) on the CPU; dist is the final LPIPS feature distance."""
    n = targets.shape[0]
    assert targets.dtype == torch.uint8
    assert targets.shape[1:] == (G.img_channels, G.img_resolution, G.img_resolution)
    assert init in ('mean', 'sample')
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    if c is None:
        c = torch.zeros([n, 0])
    c = c.to(device=device, dtype=torch.float32)
    vgg16 = torch.jit.load(vgg16_path).eval().to(device)

    # W statistics per class, from the same z for every class.
    z_stats = torch.randn([w_avg_samples, G.z_dim], generator=_generator(seed, (0,), 'cpu')).to(device)
    classes, class_of_row = torch.unique(c, dim=0, return_inverse=True)
    w_avg, w_std = [], []
    with torch.no_grad():
        for k in range(classes.shape[0]):
            w = G.mapping(z_stats, classes[k:k + 1].repeat([w_avg_samples, 1]) if G.c_dim else None)[:, 0]
            w_avg.append(w.mean(dim=0))
            w_std.append(((w - w_avg[-1]).square().sum() / w_avg_samples).sqrt())
    w_avg, w_std = torch.stack(w_avg), torch.stack(w_std)

    rows = [_generator(seed, (1, i), device) for i in range(n)]
    with torch.no_grad():
        if init == 'mean':
            w_init = w_avg[class_of_row]
        else:
            z = torch.stack([torch.randn([G.z_dim], generator=g, device=device) for g in rows])
            w_init = G.mapping(z, c if G.c_dim else None)[:, 0]

    w_out, dist_out = [], []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        gens, std = rows[lo:hi], w_std[class_of_row[lo:hi]][:, None]
        with torch.no_grad():
            target_features = _features(vgg16, targets[lo:hi].to(device).to(torch.float32), vgg_bf16)
        w_opt = w_init[lo:hi].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
        for step in range(num_steps):
            t = step / num_steps
            w_noise_scale = std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_learning_rate * lr_ramp
            w_noise = torch.stack([torch.randn([G.w_dim], generator=g, device=device) for g in gens]) * w_noise_scale
            ws = (w_opt + w_noise).unsqueeze(1).repeat([1, G.mapping.num_ws, 1])
            synth = (G.synthesis(ws, noise_mode='const') + 1) * (255 / 2)
            dist = (target_features - _features(vgg16, synth, vgg_bf16)).square().sum(dim=1)
            optimizer.zero_grad(set_to_none=True)
            dist.sum().backward()  # per-target terms: each w only receives its own gradient
            optimizer.step()
        with torch.no_grad():
            ws = w_opt.unsqueeze(1).repeat([1, G.mapping.num_ws, 1])
            synth = (G.synthesis(ws, noise_mode='const') + 1) * (255 / 2)
            dist_out.append((target_features - _features(vgg16, synth, vgg_bf16)).square().sum(dim=1).cpu())
        w_out.append(w_opt.detach().cpu())
    return torch.cat(w_out), torch.cat(dist_out)
