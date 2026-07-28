"""
Visualization of latent space trajectories over training.
Plots a t-SNE projection of latent vectors collected at different iterations,
showing how each equation moves through the space. Equations can be colored by
user-defined clusters.
"""

import argparse
import yaml
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import jpcm


def compute_embedding(
    snapshots: np.ndarray,
    shape: Tuple[int, int, int],
    method: str = "pca",
    pca_dims: int = 50,
    tsne_perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional latents to 2D for plotting.

    Args:
        snapshots: (N, D) array of flattened latent vectors
        method: "tsne", "umap", or "pca"
        pca_dims: number of PCA dimensions before t-SNE (speedup)
        tsne_perplexity: t-SNE perplexity
        random_state: randomness seed

    Returns:
        (N, 2) array of 2D coordinates
    """
    # First reduce with PCA if needed

    B, N, dim = shape

    if method == "tsne":
        if dim > pca_dims and B*N > pca_dims:
            pca = PCA(n_components=pca_dims, random_state=random_state)
            latents_reduced = pca.fit_transform(snapshots.reshape(-1, dim)).reshape((B, N, -1))[:, -1, :]
        else:
            latents_reduced = snapshots.reshape(-1, dim)
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            random_state=random_state,
            init="pca",
        )
        embedding = tsne.fit_transform(latents_reduced).reshape(B,N,2)
    elif method == "pca":
        # Just use first 2 PCA components
        pca = PCA(n_components=2, random_state=random_state)
        embedding = pca.fit(snapshots[:,-1,:]).transform(snapshots.reshape(-1, dim)).reshape(B,N,2)
    else:
        raise ValueError(f"Unknown method: {method}")
    return embedding

def plot_confusion(snapshots, strings, output="confusion.png"):
    """
    Plots confusion matrix (pairwise distances) of the latent space.
    """
    B = snapshots.shape[0]
    x = snapshots[None, :,-1,:].reshape(1,B,-1)
    d = torch.cdist(x, x, p=2.0)[0]
    nd = d / (torch.max(torch.abs(d)).detach() + 1e-8)
    plt.figure(figsize=(10,10))
    plt.imshow(torch.log(nd + 1e-8).clamp(min=-7.5).cpu().numpy() , cmap='inferno')
    plt.colorbar()
    plt.xticks(np.arange(len(strings)), strings, rotation=90)
    plt.yticks(np.arange(len(strings)), strings)
    plt.title("Confusion matrix of the latent space (distances in log)")
    plt.xlabel("Equation")
    plt.ylabel("Equation")
    plt.savefig(output)
    plt.close()
    
    
    


def plot_atlas(
    snapshots: torch.Tensor,
    strings: List[str],
    labels: List[int],
    names: List[str],
    output: str = "atlas.png",
    method: str = "pca",
    show_trajectories: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150,
) -> None:
    """Generate the semantic atlas plot.

    Args:
        snapshots: latent torch.Tensor (B, seq_len, ...)
        strings: list of RPN/equation strings, length matching B
        labels: list of integer cluster IDs, length matching B
        output: path to save the figure
        method: dimensionality reduction method ('tsne' or 'pca')
        show_trajectories: if True, draw lines connecting iterations for each
          equation
        figsize: figure size
        dpi: resolution
    """
    B, N = snapshots.shape[:2]
    snapshots = snapshots.reshape(B, N, -1)
    dim = snapshots.shape[-1]


    if len(strings) != B:
        raise ValueError(
            f"Length of strings ({len(strings)}) must match batch size B ({B})"
        )
    if labels is not None and len(labels) != B:
        raise ValueError(
            f"Length of labels ({len(labels)}) must match batch size B ({B})"
        )

    print(f"Atlas with {method} on {snapshots.shape} ")

    embedding_2d = compute_embedding(snapshots, shape=(B, N, dim), method=method).reshape(B,N,2)

    print("Embedded.")

    # Build mapping from cluster ID to color
    cluster_id_to_color: Dict[int, Tuple[float, float, float, float]] = {}
    if labels is not None:
        unique_ids = sorted(set(labels))
        clusters = max(labels) + 1

        # Support both older and newer matplotlib versions for getting colormaps
        cmap = jpcm.get('fuyu').resampled(clusters + 1)
        for i, cid in enumerate(unique_ids):
            cluster_id_to_color[cid] = cmap(1 + i % clusters)

    # Plot Setup
    plt.figure(figsize=figsize)

    # Iterate through each equation trajectory
    for idx in range(B):
        rpn = strings[idx]
        points_arr = embedding_2d[idx]  # Shape: (seq_len, 2)

        # Assign style based on cluster labels
        if labels is not None:
            cid = labels[idx]
            color = cluster_id_to_color.get(cid, "gray")
            alpha = 0.8
            linewidth = 0.5
        else:
            color = "gray"
            alpha = 0.4
            linewidth = 1.0

        # Draw trajectory lines
        if show_trajectories and N > 1:
            plt.plot(
                points_arr[:, 0],
                points_arr[:, 1],
                "-",
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

        # Mark start point (Iteration 0)
        # plt.scatter(
        #     points_arr[0, 0],
        #     points_arr[0, 1],
        #     color=color,
        #     s=120,
        #     alpha=alpha,
        #     marker="o",
        #     edgecolors="white",
        #     linewidth=1.0,
        #     zorder=2,
        # )
        # Mark end point (Final Iteration)
        plt.scatter(
            points_arr[-1, 0],
            points_arr[-1, 1],
            color=color,
            s=250,
            alpha=1.0,
            marker="*",
            edgecolors="white",
            linewidth=1.5,
            zorder=3,
        )

    # Legend for cluster IDs
    if cluster_id_to_color:
        legend_elements = [
            Patch(facecolor=cluster_id_to_color[cid], label=f"{names[cid]}")
            for cid in sorted(cluster_id_to_color.keys())
        ]
        plt.legend(handles=legend_elements, loc="best", framealpha=0.9)

    plt.title("Clustering trajectories in a latent atlas")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    print(f"Atlas saved to {output}")