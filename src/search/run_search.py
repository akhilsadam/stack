from __future__ import annotations

import os
import sys
import torch
import yaml
from tqdm import tqdm

# ── paths (relative to this file's location: vectorspace/src/) ──────
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# _PKG resolves to /path/to/vectorspace/src/
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# from search.SGD_RWR import Search
from search.SGD_RWRv2 import Search
# from search.SGD_GMvD import Search
# from search.SGD_LGM import Search

# from space.tokenizer import Tokenizer
# from search.SGD_token import Search

from space.tokenizer import BasicTokenizer as Tokenizer
# from search.SGD_LRWR import Search
# from search.SGD_LRWRv5 import Search
# from search.SGD_RWR import Search
# from search.SGD import Search


from vis import SpaceVis as _vis
from apps.qg_basic._eval import build_vocab, QGEvaluator

_PROBLEM_CFG = os.path.join(_PKG, "apps/qg_basic/problem_config.yaml")
_CONFIG      = os.path.join(_PKG, "apps/qg_basic/config.yaml")
_CLUSTERS    = os.path.join(_PKG, "apps/qg_basic/clusters.yaml")
_CHECKPOINT  = None  # override via env SEARCH_CKPT or --checkpoint


def main():
    # ── load configs ───────────────────────────────────────────────
    with open(_PROBLEM_CFG) as f:
        problem_cfg = yaml.safe_load(f)
    with open(_CONFIG) as f:
        train_cfg = yaml.safe_load(f)

    torch.manual_seed(train_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Package root: {_PKG}")

    # ── build components ───────────────────────────────────────────
    print("Building vocabulary and QG evaluator …")
    vocab = build_vocab()
    evaluator = QGEvaluator(
        grid_size=train_cfg.get("grid_size", 64),
        seed=train_cfg.get("seed", 42),
        **problem_cfg,
    )
    print(f"Initialising Tokenizer "
          f"(batch={train_cfg.get('batch')}, "
          f"seq_len={train_cfg.get('seq_len')}, "
          f"dim={train_cfg.get('dim')}) …")
    tokenizer = Tokenizer(
        vocab=vocab,
        _eval=evaluator,
        vis=_vis(_CLUSTERS),
        **train_cfg,
    )

    # optionally load a checkpoint
    ckpt = os.environ.get("SEARCH_CKPT") or _CHECKPOINT
    if ckpt and os.path.isfile(ckpt):
        print(f"Loading checkpoint from {ckpt} …")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        tokenizer.load_state_dict(state)
        tokenizer.eval()
    else:
        print("No checkpoint — using freshly initialised tokenizer.")

    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    # ── search ─────────────────────────────────────────────────────
    init_rpn   = problem_cfg.get("PDE_init", problem_cfg["PDE"])
    target_rpn = problem_cfg["PDE"]
    search_cfg = train_cfg.get("search", {})

    # sgd = SGD(
    #     tokenizer,
    #     evaluator=evaluator,
    #     steps=search_cfg.get("steps", 200),
    #     pop_size=search_cfg.get("pop_size", 64),
    #     noise_std=search_cfg.get("noise_std", 0.5),
    #     lr=search_cfg.get("lr", 1e-1),
    #     log_every=search_cfg.get("log_every", 20),
    # )

    sgd = Search(
        tokenizer,
        evaluator=evaluator,
    )


    print(f"\n{'='*60}")
    print(f"Target PDE:  {target_rpn}")
    print(f"Initial PDE: {init_rpn}")
    print(f"Steps: {sgd.steps}  |  Pop size: {sgd.pop_size}  "
          f"|  Noise σ: {sgd.noise_std}  |  lr: {sgd.lr}")
    print(f"{'='*60}\n")

    best_rpn, best_loss = sgd.search(init_rpn, target_pde=target_rpn)

    # ── results ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Search finished.")
    print(f"Best RPN:  {best_rpn}")
    print(f"Best loss: {best_loss:.6e}")
    LaTeX_t = evaluator.to_latex(target_rpn)
    LaTeX_b = evaluator.to_latex(best_rpn)
    print(f"Target: ${LaTeX_t}$")
    print(f"Best:   ${LaTeX_b}$")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()