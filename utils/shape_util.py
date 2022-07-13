# The geodesic distance computation is adapted from
# https://github.com/pvnieo/SURFMNet-pytorch/blob/master/surfmnet/preprocess.py

import os
import numpy as np
import trimesh
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors


def compute_geodesic_matrix(verts, faces, NN=500):
    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), 'Graph not connected'
    vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increase NN.')
    return geodesic_x


def read_off(file):
    with open(file, 'r') as f:
        if f.readline().strip() != "OFF":
            raise "Not a valid OFF header"

        n_verts, n_faces, _ = tuple([int(s) for s in f.readline().strip().split(" ")])
        verts = [[float(s) for s in f.readline().strip().split(" ")] for i_vert in range(n_verts)]
        faces = [[int(s) for s in f.readline().strip().split(" ")][1:] for i_face in range(n_faces)]

    return np.array(verts), np.array(faces)


def write_off(file, verts, faces):
    with open(file, 'w') as f:
        f.write("OFF\n")
        f.write(f"{verts.shape[0]} {faces.shape[0]} {0}\n")
        for x in verts:
            f.write(f"{' '.join(map(str, x))}\n")
        for x in faces:
            f.write(f"{len(x)} {' '.join(map(str, x))}\n")


def read_txt(file):
    verts = np.loadtxt(file)

    return verts, None


def read_mesh(file):
    """
    Read mesh from file.

    Args:
        file (str): file name

    Returns:
        verts (np.ndarray): vertices [V, 3]
        faces (np.ndarray): faces [F, 3]
    """
    if os.path.splitext(file)[1] == ".off":
        return read_off(file)
    elif os.path.splitext(file)[1] == ".txt":
        return read_txt(file)
    else:
        raise f"File extention {os.path.splitext(file)[1]} not implemented yet!"
