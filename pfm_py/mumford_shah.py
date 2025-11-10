import torch

from pfm_py.optimize_indicators import VStepOptions

ALMOST_ZERO = 1e-12

def mumford_shah_cost_unvectorized(v, vertices, triangles, perturb, opts : VStepOptions):
    var = 2 * opts.tv_sigma**2
    # Called "h" in the original code
    # The functional xi(v) := kronecker_delta(eta(v) - 0.5) is smoothly approximated
    xi = torch.exp(-(v - opts.tv_mean)**2 / var)
    # Optional weighting of vertices (currently perturb is set to ones in optimize_v())
    xi = xi * perturb

    cost = 0.0
    for triangle in triangles:
        i, j, k = triangle
        v1, v2, v3 = v[i], v[j], v[k]
        v_alpha = v2 - v1
        v_beta = v3 - v1

        x1, x2, x3 = vertices[i], vertices[j], vertices[k]
        e1 = x2 - x1
        e2 = x3 - x1
        E, F, G = torch.dot(e1, e1), torch.dot(e1, e2), torch.dot(e2, e2)
        norm_grad_v = torch.sqrt(v_alpha**2 * G - 2 * v_alpha * v_beta * F + v_beta**2 * E + ALMOST_ZERO)

        if opts.mumford_shah_area_weighted:
            det = torch.sqrt(torch.abs(E * G - F * F) + ALMOST_ZERO)
            norm_grad_v /= det

        xi_sum = xi[i] + xi[j] + xi[k]
        if xi_sum > ALMOST_ZERO:
            cost += xi_sum * norm_grad_v / 6.0

    return cost

def mumford_shah_cost(v, vertices, triangles, perturb, opts: VStepOptions):
    # Optimization potential: Precompute E, F, G (they depend only on the geometry, not on v)

    var = 2 * opts.tv_sigma**2
    # Called "h" in the original code
    # The functional xi(v) := kronecker_delta(eta(v) - 0.5) is smoothly approximated
    xi = torch.exp(-(v - opts.tv_mean)**2 / var)
    # Optional weighting of vertices (currently perturb is set to ones in optimize_v())
    xi = xi * perturb

    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    x1, x2, x3 = vertices[i], vertices[j], vertices[k]
    e1, e2 = x2 - x1, x3 - x1

    E = torch.sum(e1 * e1, dim=1)
    F = torch.sum(e1 * e2, dim=1)
    G = torch.sum(e2 * e2, dim=1)

    v1, v2, v3 = v[i], v[j], v[k]
    v_alpha, v_beta = v2 - v1, v3 - v1
    norm_grad_v = torch.sqrt(v_alpha**2 * G - 2 * v_alpha * v_beta * F + v_beta**2 * E + ALMOST_ZERO)

    if opts.mumford_shah_area_weighted:
        det = torch.sqrt(torch.abs(E * G - F * F) + ALMOST_ZERO)
        norm_grad_v /= det

    xi_sum = xi[i] + xi[j] + xi[k]
    mask = xi_sum > ALMOST_ZERO
    cost = torch.sum(xi_sum[mask] * norm_grad_v[mask]) / 6.0
    return cost