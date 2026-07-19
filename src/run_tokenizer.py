import os
import torch
import yaml
from tokenizer import Tokenizer
from apps.qg._eval import build_vocab, QGEvaluator
from vis import SpaceVis as _vis

config_path = "apps/qg/config.yaml"
cluster_path = "apps/qg/clusters.yaml"

def main():
    print(os.getcwd())

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config.get("seed", 42))
    print("Building vocabulary and QG Evaluator...")
    vocab = build_vocab()
    evaluator = QGEvaluator(
        grid_size=config.get("grid_size", 64), 
        seed=config.get("seed", 42)
    )

    print(f"Initializing Tokenizer (batch_size={config.get('batch')}, seq_len={config.get('seq_len')}, dim={config.get('dim')}, iter={config.get('iter')})...")
    
    tokenizer = Tokenizer(
        vocab=vocab,
        _eval=evaluator,
        batch=config.get("batch", 64),
        seq_len=config.get("seq_len", 12),
        dim=config.get("dim", 8),
        depth=config.get("depth", 2),
        steps=config.get("steps", 2),
        lr=float(config.get("lr", 1e-3)),
        _iter=config.get("iter", 2000),
        vis=_vis(cluster_path)
    )

    print("Training finished successfully.")


if __name__ == "__main__":
    main()