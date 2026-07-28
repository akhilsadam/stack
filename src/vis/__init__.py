# Visualization module for vectorspace
# Provides atlas plotting for latent space trajectories                                                                                                                                          

import os
import yaml
from typing import List, Dict, Optional
import torch
from .atlas import plot_atlas, plot_confusion


class SpaceVis:
    """Manages snapshot collection and atlas plotting for vector space."""

    def __init__(self, config_path: str):
        """Initialize visualization space from a YAML config.

        Args:
            config_path: Path to YAML config (typically apps/qg/clusters.yaml).
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}
        self.clusters = self.config.get('clusters', [])                                                                                                                                                               
                                                                                                                                                                    
        self.vis = self.config.get('vis', False)
        self.vis_every = self.config.get('vis_every', 0)
        self.vis_folder = self.config.get('vis_folder', 'vis')  

        # Build canonical strings list and cluster label list
        self._strings = [
            eq for cluster in self.clusters
            for eq in cluster.get('strings', [])
        ]    
        
        self._labels = [
            i for i, cluster in enumerate(self.clusters)
            for _ in cluster.get('strings', [])
        ]

        self._names = [
            cluster.get('name', str(i)) for i, cluster in enumerate(self.clusters)
        ]

        self.snapshots = []

        if self.vis:
            os.makedirs(self.vis_folder, exist_ok=True)

    def snapshot(self, it: int, embed):
        """Add a snapshot of latent vectors for the given iteration.
        """

        if it % self.vis_every != 0:
            return self._strings
        
        z = embed(self._strings)
        self.snapshots.append(z.detach())
        return self._strings

    def plot(self, it, **kwargs) -> str:
        output_path = os.path.join(self.vis_folder, f'atlas_{it:3d}.png')
        coutput_path = os.path.join(self.vis_folder, f'confusion_{it:3d}.png')
        snapshots = torch.stack(self.snapshots, dim=1).cpu().clone()
        plot_confusion(snapshots, self._strings, output=coutput_path)
        # plot_atlas(              
        #     snapshots,
        #     self._strings,                                                                                                                                                                                                 
        #     self._labels,
        #     self._names,
        #     output=output_path,
        #     **kwargs
        # )
        return output_path

    #   def save_snapshots(self, path: Optional[str] = None) -> str:
    #       """Save raw snapshot data to a torch .pt file.                                                                                                                                                                

    #       Args:
    #           path: File path. Defaults to 'vis/snapshots.pt' within vis_folder.

    #       Returns:
    #           Path where snapshots were saved.
    #       """
    #       if path is None:
    #           path = os.path.join(self.vis_folder, 'snapshots.pt')
    #       torch.save({'snapshots': self.snapshots}, path)
    #       return path
