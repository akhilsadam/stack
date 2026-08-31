from __future__ import annotations

import os
import sys
import torch
import yaml
from tqdm import tqdm

# TODO: remove sys path nonsense
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from space.tokenizer import Stack, Operator
# from MCTS import MCTSStack as Stack, PUCTOperator as Operator
# from search.SGD import Search

app = "equation_v0_linear"
# app = "equation_v1_branched"
# app = "qg_basic"


import importlib.util
spec = importlib.util.spec_from_file_location("eval", f"apps/{app}/_eval.py")
eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval)

build_vocab = eval.build_vocab
Evaluator   = eval.Evaluator
_CONFIG     = os.path.join(_PKG, f"apps/{app}/config.yaml")


def main():
    # ── load configs ───────────────────────────────────────────────
    with open(_CONFIG) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Package root: {_PKG}")

    # ── build components ───────────────────────────────────────────
    print("Building vocabulary and QG evaluator …")
    vocab = build_vocab(_Operator=Operator, **cfg)
    
    tokenizer = Stack(
        init_str = cfg.get('init'),
        vocab=vocab,
        **cfg,
    )

    tokenizer = tokenizer.to(device)
    tokenizer._train(Evaluator(**cfg))

    # # ── search ─────────────────────────────────────────────────────
    # init_rpn   = cfg.get("PDE_init", cfg["PDE"])
    # target_rpn = cfg["PDE"]

    # print(f"\n{'='*60}")
    # print(f"Target PDE:  {evaluator.target_pde}")
    # print(f"Base PDE:    {evaluator.base_pde}")
    # print(f"Initial Term: {init_rpn}")
    # print(f"Steps: {sgd.steps} with lr: {sgd.lr}")
    # print(f"{'='*60}\n")

    # best_rpn, best_loss = sgd.search(init_rpn)

    # # ── results ────────────────────────────────────────────────────
    # print(f"\n{'='*60}")
    # print("Search finished.")
    # print(f"Target PDE:  {evaluator.target_pde}")
    # print(f"Base PDE:    {evaluator.base_pde}")
    # print(f"Initial Term: {init_rpn}")
    # print(f"Steps: {sgd.steps}  |  Pop size: {sgd.pop_size}  "
    #       f"|  Noise σ: {sgd.noise_std}  |  lr: {sgd.lr}")
    # print(f"\n{'-'*60}")
    # print(f"Best Term:  {best_rpn}")
    # print(f"Best loss: {best_loss:.6e}")
    # LaTeX_t = evaluator.to_latex(target_rpn)
    # LaTeX_b = evaluator.to_latex(best_rpn)
    # print(f"Target: ${LaTeX_t}$")
    # print(f"Best:   ${LaTeX_b}$")
    # print(f"{'='*60}")


if __name__ == "__main__":
    main()