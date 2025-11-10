import numpy as np
from pfm_py import mesh_utils as mu
from pfm_py.fem import mass_matrix_barycentric, mass_matrix_full
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

# test fem mass matrix functions
# Simple tetrahedron fragment for testing
V = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
F = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ])

# Compute total mesh area
total_area = mu.tri_areas(V, F).sum()

# Build both mass matrices
M_bary = mass_matrix_barycentric(V, F)    
M_full = mass_matrix_full(V, F)

print("Total area:", total_area)
print("Trace(M_bary):", M_bary.diagonal().sum())
print("Trace(M_full):", M_full.diagonal().sum())

# The trace of the mass matrix should roughly equal the total area
# (since ∫1 dA = total area, and <1,1> = 1^T M 1 = trace(M))
print("1^T M_bary 1 =", np.ones(len(V)) @ (M_bary @ np.ones(len(V))))
print("1^T M_full 1 =", np.ones(len(V)) @ (M_full @ np.ones(len(V))))

# Check positive definiteness (all diagonal entries > 0)
print("Min diag(bary) =", M_bary.diagonal().min())
print("Min diag(full) =", M_full.diagonal().min())

print("Test passed if all values are positive and total area ≈ trace(M).")