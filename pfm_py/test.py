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

from dataclasses import dataclass

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

partial_folders = ["cuts", "holes"]
for folder in partial_folders:
    partial_files = os.listdir(data_path + "/SHREC16/" + folder + "/off")
    i = 0
    for partial_file in partial_files:
        # remove extension safely
        partial_mesh_name = os.path.splitext(partial_file)[0]

        # safe extraction of the full mesh name from the partial's filename
        parts = partial_mesh_name.split('_')
        if len(parts) >= 2:
            full_mesh_name = parts[1]
        else:
            full_mesh_name = partial_mesh_name
        mesh_data = main.TestMeshData(
            name=partial_mesh_name,
            full_mesh=data_path + f"/SHREC16/null/off/{full_mesh_name}.off",
            partial_mesh=data_path + f"/SHREC16/{folder}/off/{partial_file}",
            ground_truth=data_path + f"/SHREC16/{folder}/corres/{partial_mesh_name}.vts"
        )
        result_path = f"{target_path}/{folder}/{partial_mesh_name}"

        if partial_mesh_name != "cuts_cat_shape_10":
            continue

        # Vary C_max_iter with step size 500
        c_max_iter_values = list(range(5000, 5001, 500))
        mean_errors = []
        
        opts.descriptor_type = 'shot'
        opts.early_stopping = False
        
        for c_max_iter in c_max_iter_values:
            opts.C_max_iter = c_max_iter
            result = main.run(mesh_data, output_folder=None, opts=opts)
            mean_error = result
            mean_errors.append(mean_error)
            print(f"C_max_iter = {c_max_iter}: mean geodesic error = {mean_error:.6f}")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(c_max_iter_values, mean_errors, marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.xlabel('C_max_iter', fontsize=12)
        plt.ylabel('Mean Geodesic Error', fontsize=12)
        plt.title(f'Mean Geodesic Error vs C_max_iter\nMesh: {partial_mesh_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"{target_path}/c_iter_cuts_cat_shape_10_test3.png"
        os.makedirs(target_path, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")
        plt.show()


        # opts.descriptor_type = 'shot'
        # opts.C_max_iter = 1500
        # opts.early_stopping = False
        # error = main.run(mesh_data, output_folder=None, opts=opts)
        # print(f"Mean geodesic error with early stopping: {error:.6f}")
