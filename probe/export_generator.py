"""Write a pickle holding only G_ema, so a consumer can load it without D's and the augment pipe's deps (scipy).

Run in this repo's env:  python probe/export_generator.py SNAPSHOT.pkl OUT.pkl
Load elsewhere (this repo on the path):  pickle.load(f)["G_ema"]
"""
import pickle
import sys

import legacy

with open(sys.argv[1], "rb") as f:
    G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).cpu()
with open(sys.argv[2], "wb") as f:
    pickle.dump({"G_ema": G}, f)
print(f"G_ema: res {G.img_resolution}, c_dim {G.c_dim}, w_dim {G.w_dim}, num_ws {G.mapping.num_ws} -> {sys.argv[2]}")
