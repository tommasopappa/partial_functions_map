from typing import Iterator
import os

from pfm_py.dataset.mesh_pair import MeshPair

class Shrec16:
    """Iterator that yields MeshPair objects for all full-partial pairs in the SHREC'16 dataset."""
    
    def __init__(self, data_path: str, partial_folders: list = ["cuts", "holes"]):
        self.data_path = data_path
        self.partial_folders = partial_folders
    
    def __iter__(self) -> Iterator[MeshPair]:
        """Iterate over all samples across all partial folders"""
        for folder in self.partial_folders:
            partial_files = os.listdir(os.path.join(self.data_path, folder, "off"))
            
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
                yield MeshPair(
                    name=partial_mesh_name,
                    full_mesh=os.path.join(self.data_path, "null", "off", f"{full_mesh_name}.off"),
                    partial_mesh=os.path.join(self.data_path, folder, "off", partial_file),
                    ground_truth=os.path.join(self.data_path, folder, "corres", f"{partial_mesh_name}.vts")
                )