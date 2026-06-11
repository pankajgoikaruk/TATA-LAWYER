import json, os, time
from pathlib import Path


def make_run_dir(root):
    ts = time.strftime('%Y%m%d_%H%M%S')
    rd = Path(root) / ts
    rd.mkdir(parents=True, exist_ok=True)
    (rd / 'ckpt').mkdir(exist_ok=True)
    return str(rd)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)