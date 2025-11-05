import numpy as np
from pfm_py import mesh_utils as mu
# test mesh_utils functions
V = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
F = np.array([[0,1,2],[0,1,3]])

print("Face areas:", mu.tri_areas(V,F))
print("Total area:", mu.tri_areas(V,F).sum())
print("Vertex areas:", mu.vertex_areas_barycentric(V,F))
V_center = mu.center_vertices(V)
V_scaled, s = mu.normalize_unit_area(V_center, F)
print("Scale factor:", s)
print("Sum area after scaling:", mu.tri_areas(V_scaled,F).sum())
I, J = mu.vertex_vertex_adjacency(F)
print("Edges:", np.vstack([I,J]).T)
V2, F2, idx = mu.remove_isolated_vertices(V, F)
print("After removing isolated vertices:", V2.shape, F2.shape)