import os
import scipy.io as sio
import numpy as np

import torch

from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
try:
    from pyshot import get_descriptors
except:
    print('For SHOT descriptor computation, it requires to install pyshot.')
    exit(0)
from utils.geometry_util import get_operators, laplacian_decomposition
from utils.shape_util import read_mesh, compute_geodesic_matrix, write_off


# SHOT's hyper-parameters
RADIUS = 0.09
NUM_BINS = 10
MIN_NEIGHBORS = 3


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser('Preprocess .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--n_eig', type=int, default=150, help='number of eigenvectors/values to compute.')
    parser.add_argument('--no_eig', action='store_true', help='no laplacian eigen-decomposition')
    parser.add_argument('--no_shot', action='store_true', help='no shot descriptor.')
    parser.add_argument('--no_dist', action='store_true', help='no geodesic matrix.')
    parser.add_argument('--no_diffusion', action='store_true', help='no diffusion spectral operators.')
    args = parser.parse_args()

    # sanity check
    data_root = args.data_root
    n_eig = args.n_eig
    no_eig = args.no_eig
    no_shot = args.no_shot
    no_dist = args.no_dist
    no_diffusion = args.no_diffusion
    assert n_eig > 0, f'Invalid n_eig: {n_eig}'
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'

    # make directories.
    if not no_eig:
        spectral_dir = os.path.join(data_root, 'spectral')
        os.makedirs(spectral_dir, exist_ok=True)

    if not no_diffusion:
        diffusion_dir = os.path.join(data_root, 'diffusion')
        os.makedirs(diffusion_dir, exist_ok=True)

    if not no_dist:
        dist_dir = os.path.join(data_root, 'dist')
        os.makedirs(dist_dir, exist_ok=True)

    if not no_shot:
        shot_dir = os.path.join(data_root, 'shot')
        os.makedirs(shot_dir, exist_ok=True)

    # read .off files
    off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))
    assert len(off_files) != 0

    for off_file in tqdm(off_files):
        verts, faces = read_mesh(off_file)
        filename = os.path.basename(off_file)

        if not no_eig:
            # recompute laplacian decomposition
            evals, evecs, evecs_trans, sqrt_area = laplacian_decomposition(verts, faces, k=n_eig)
            # save results
            mat = {'evals': evals, 'evecs': evecs, 'evecs_trans': evecs_trans}
            sio.savemat(os.path.join(spectral_dir, filename.replace('.off', '.mat')), mat)

        if not no_shot:
            # compute shot descriptor
            shot_features = get_descriptors(verts, faces,
                                            radius=RADIUS, local_rf_radius=RADIUS,
                                            min_neighbors=MIN_NEIGHBORS,
                                            n_bins=NUM_BINS,
                                            double_volumes_sectors=True,
                                            use_interpolation=True,
                                            use_normalization=True).reshape(-1, 352)

            # save results
            mat = {'shot': shot_features}
            sio.savemat(os.path.join(shot_dir, filename.replace('.off', '.mat')), mat)

        if not no_diffusion:
            # compute and save spectral operators for diffusion net
            spectral_ops = get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(), k=128,
                                         cache_dir=diffusion_dir)

        if not no_dist:
            # compute distance matrix
            dist_mat = compute_geodesic_matrix(verts, faces)

            # save results
            sio.savemat(os.path.join(dist_dir, filename.replace('.off', '.mat')), {'dist': dist_mat})
