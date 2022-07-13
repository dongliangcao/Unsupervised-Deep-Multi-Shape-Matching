import numpy as np
import torch
from torch_geometric.nn import knn


def nn_query(feat_x, feat_y):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].

    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    p2p = knn(feat_x, feat_y, k=1)[1].flatten()
    return p2p


def fmap2pointmap(C12, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        C12: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    return nn_query(torch.matmul(evecs_x, C12.t()), evecs_y)


def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Convert a point-to-point map to functional map

    Args:
        p2p (np.ndarray): point-to-point map (shape x -> shape y). [Vx]
        evecs_x (np.ndarray): eigenvectors of shape x. [Vx, K]
        evecs_y (np.ndarray): eigenvectors of shape y. [Vy, K]
    Returns:
        C21 (np.ndarray): functional map (shape y -> shape x). [K, K]
    """
    C21 = torch.linalg.lstsq(evecs_x, evecs_y[p2p, :]).solution
    return C21


def refine_pointmap_icp(p2p, evecs_x, evecs_y, n_iters=10):
    """
    Regular Iterative Closest Point (ICP) to refine a point-to-point map
    Args:
        p2p: point-to-point map: shape x -> shape y. [Vx]
        evecs_x: eigenvectors of shape x. [Vx, K]
        evecs_y: eigenvectors of shape y. [Vy, K]
        n_iters (int, optional): number of iterations. Default 10.
    """
    p2p_refined = p2p
    for _ in range(n_iters):
        C21_refined = pointmap2fmap(p2p_refined, evecs_x, evecs_y)
        p2p_refined = fmap2pointmap(C21_refined, evecs_y, evecs_x)

    return p2p_refined, C21_refined


def refine_pointmap_zoomout(p2p, evecs_x, evecs_y, k_start, n_iters=10):
    """
    ZoomOut to refine a point-to-point map
    Args:
        p2p: point-to-point map: shape x -> shape y. [Vx]
        evecs_x: eigenvectors of shape x. [Vx, K]
        evecs_y: eigenvectors of shape y. [Vy, K]
        k_start (int): number of eigenvectors to start
        n_iters (int, optional): number of iterations. Default 10.
    """
    k_end = evecs_x.shape[1]
    step = (k_end - k_start) // n_iters
    inds = np.arange(k_start, k_end + step, step)

    p2p_refined = p2p
    for i in inds:
        C21_refined = pointmap2fmap(p2p_refined, evecs_x[:, :i], evecs_y[:, :i])
        p2p_refined = fmap2pointmap(C21_refined, evecs_y[:, :i], evecs_x[:, :i])

    return p2p_refined, C21_refined
