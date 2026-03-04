from dataclasses import dataclass

@dataclass
class MeshPair:
	name: str
	full_mesh: str
	partial_mesh: str
	ground_truth: str
