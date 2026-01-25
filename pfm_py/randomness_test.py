from asyncio import run
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options
import pfm_py.main as main

import torch
import open3d as o3d
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt

from typing import Iterator

class SampleIterator:
    """Iterator that yields TestMeshData objects for all partial meshes"""
    
    def __init__(self, data_path: str, target_path: str, partial_folders: list):
        self.data_path = data_path
        self.target_path = target_path
        self.partial_folders = partial_folders
    
    def __iter__(self) -> Iterator['main.TestMeshData']:
        """Iterate over all samples across all partial folders"""
        for folder in self.partial_folders:
            partial_files = os.listdir(os.path.join(self.data_path, "SHREC16", folder, "off"))
            
            for partial_file in partial_files:
                # Remove extension safely
                partial_mesh_name = os.path.splitext(partial_file)[0]
                
                # Safe extraction of the full mesh name from the partial's filename
                parts = partial_mesh_name.split('_')
                if len(parts) >= 2:
                    full_mesh_name = parts[1]
                else:
                    full_mesh_name = partial_mesh_name
                
                # Create and yield mesh data
                yield main.TestMeshData(
                    name=partial_mesh_name,
                    full_mesh=os.path.join(self.data_path, "SHREC16", "null", "off", f"{full_mesh_name}.off"),
                    partial_mesh=os.path.join(self.data_path, "SHREC16", folder, "off", partial_file),
                    ground_truth=os.path.join(self.data_path, "SHREC16", folder, "corres", f"{partial_mesh_name}.vts")
                )

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description='Partial Functions Map - 3D shape matching',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=""
)

parser.add_argument(
    '--data-path',
    type=str,
    default='/usr/prakt/w0010/SAVHA/shape_data',
    help='Path to the shape data directory (default: /usr/prakt/w0010/SAVHA/shape_data)'
)
parser.add_argument(
    '--target-path',
    type=str,
    default='results',
    help='Path to the output results directory (default: results)'
)

args = parser.parse_args()

# Data path
data_path = args.data_path
target_path = args.target_path
state_path = os.path.join(target_path, 'state.txt')

print(f"Data path: {data_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
opts = Options(device)

sample_iterator = SampleIterator(data_path, target_path, ["cuts", "holes"]).__iter__()
sample1 = next(sample_iterator)

opts.descriptor_type = 'dinov3'
opts.early_stopping = False
opts.C_max_iter = 10
opts.v_max_iter = 10
opts.max_outer_iter = 1

err1 = main.run(sample1, output_folder=None, opts=opts)
err2 = main.run(sample1, output_folder=None, opts=opts)

print()
print(f"Error for {sample1.name}: {err1}")
print(f"Error for {sample1.name}: {err2}")