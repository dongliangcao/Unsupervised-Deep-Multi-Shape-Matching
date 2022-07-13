import os
import numpy as np
import scipy.io as sio
from itertools import product
from glob import glob

import torch
from torch.utils.data import Dataset

from utils.shape_util import read_mesh
from utils.registry import DATASET_REGISTRY


class PairShapeDataset(Dataset):
    """
    Pair Shape Dataset
    """

    def __init__(self,
                 data_root,
                 return_vertices=True,
                 return_shot=True, num_shot=352,
                 return_evecs=True, num_evecs=120,
                 return_label=False, return_dist=False):
        """
        Pair Shape Dataset

        Args:
            data_root (str): Data root.
            return_vertices (bool, optional): Indicate whether return vertices. Default True.
            return_shot (bool, optional): Indicate whether return SHOT descriptor. Default True.
            num_shot (int, optional): Number of SHOT descriptor to return, only valid for return_shot=True. Default 352.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return, only valid for return_evecs=True. Default 120.
            return_label (bool, optional): Indicate whether return label corresponds to reference shape. Default False.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        # sanity check
        assert return_vertices or return_shot or return_evecs or return_dist, \
            '"return_vertices", "return_shot", "return_evecs", "return_dist" must at least one be true'
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_vertices = return_vertices
        self.return_shot = return_shot
        self.return_evecs = return_evecs
        self.return_label = return_label
        self.return_dist = return_dist
        self.num_shot = num_shot
        self.num_evecs = num_evecs

        self.off_files = None
        self.shot_files = None
        self.evecs_files = None
        self.corr_files = None
        self.dist_files = None

        # check the data path contains .vts files
        corr_path = os.path.join(data_root, 'corres')
        assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
        self.corr_files = sorted(glob(f'{corr_path}/*.vts'))

        # check the data path conatins .off files
        if return_vertices:
            off_path = os.path.join(data_root, 'off')
            assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
            self.off_files = sorted(glob(f'{off_path}/*.off'))

        # check the shot path contains .mat files
        if return_shot:
            shot_path = os.path.join(data_root, 'shot')
            assert os.path.isdir(shot_path), f'Invalid path {shot_path} not containing .mat files'
            self.shot_files = sorted(glob(f'{shot_path}/*.mat'))

        # check the spectral path contains .mat files
        if return_evecs:
            evecs_path = os.path.join(data_root, 'spectral')
            assert os.path.isdir(evecs_path), f'Invalid path {evecs_path} not containing .mat files'
            self.evecs_files = sorted(glob(f'{evecs_path}/*.mat'))

        # check the data path contains .mat files
        if return_dist:
            dist_path = os.path.join(data_root, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not conatining .mat files'
            self.dist_files = sorted(glob(f'{dist_path}/*.mat'))

        # sanity check
        self._size = len(self.corr_files)
        assert self._size != 0
        if return_vertices:
            assert self._size == len(self.off_files)

        if return_shot:
            assert self._size == len(self.shot_files)

        if return_evecs:
            assert self._size == len(self.evecs_files)

        if return_dist:
            assert self._size == len(self.dist_files)

        # compute combinations
        self._combinations = list(product(range(self._size), repeat=2))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self._combinations[index]
        # get first shape data
        first_data = self._load(first_index)
        # get second shape data
        second_data = self._load(second_index)

        return {'first': first_data, 'second': second_data}

    def _load(self, index):
        item = dict()

        # get vertices
        if self.return_vertices:
            off_file = self.off_files[index]
            verts, faces = read_mesh(off_file)
            item['verts'] = torch.from_numpy(verts).float()
            item['faces'] = torch.from_numpy(faces).long()

        # get SHOT descriptor
        if self.return_shot:
            mat = sio.loadmat(self.shot_files[index])
            item['shot'] = torch.from_numpy(mat['shot'][:, :self.num_shot]).float()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            mat = sio.loadmat(self.evecs_files[index])
            item['evecs'] = torch.from_numpy(mat['evecs'][:, :self.num_evecs]).float()
            item['evecs_trans'] = torch.from_numpy(mat['evecs_trans'][:self.num_evecs]).float()
            item['evals'] = torch.from_numpy(mat['evals'][:self.num_evecs].flatten()).float()

        # get geodesic distance matrix
        if self.return_dist:
            mat = sio.loadmat(self.dist_files[index])
            item['dist'] = torch.from_numpy(mat['dist']).float()

        # get gt correspondence
        if self.return_label:
            corrs = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from zero
            item['sample'] = torch.from_numpy(corrs).long()

        return item

    def __len__(self):
        return len(self._combinations)


@DATASET_REGISTRY.register()
class PairFaustDataset(PairShapeDataset):
    """
    Pair Faust Dataset
    """

    def __init__(self, data_root,
                 phase, return_vertices=True,
                 return_shot=True, num_shot=352,
                 return_evecs=True, num_evecs=120,
                 return_label=True, return_dist=False):
        super(PairFaustDataset, self).__init__(data_root, return_vertices,
                                               return_shot, num_shot,
                                               return_evecs, num_evecs,
                                               return_label, return_dist)
        assert phase in ['train', 'val', 'full'], f'Invalid phase {phase}, only "train" or "val" or "full"'
        assert self._size == 100, f'Expected FAUST has 100 shape files, but get {self._size} files.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.evecs_files:
                self.evecs_files = self.evecs_files[:80]
            if self.shot_files:
                self.shot_files = self.shot_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
            if self.dist_files:
                self.dist_files = self.dist_files[:80]
            self._size = 80
        elif phase == 'val':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.evecs_files:
                self.evecs_files = self.evecs_files[80:]
            if self.shot_files:
                self.shot_files = self.shot_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
            if self.dist_files:
                self.dist_files = self.dist_files[80:]
            self._size = 20

        # reset combinations
        self._combinations = list(product(range(self._size), repeat=2))


@DATASET_REGISTRY.register()
class PairScapeDataset(PairShapeDataset):
    """
    Pair SCAPE Dataset
    """

    def __init__(self, data_root,
                 phase, return_vertices=True,
                 return_shot=True, num_shot=352,
                 return_evecs=True, num_evecs=120,
                 return_label=True, return_dist=False):
        super(PairScapeDataset, self).__init__(data_root, return_vertices,
                                               return_shot, num_shot,
                                               return_evecs, num_evecs,
                                               return_label, return_dist)
        assert phase in ['train', 'val', 'full'], f'Invalid phase {phase}, only "train" or "val" or "full"'
        assert self._size == 71, f'Expected SCAPE has 71 shape files, but get {self._size} files.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:51]
            if self.evecs_files:
                self.evecs_files = self.evecs_files[:51]
            if self.shot_files:
                self.shot_files = self.shot_files[:51]
            if self.corr_files:
                self.corr_files = self.corr_files[:51]
            if self.dist_files:
                self.dist_files = self.dist_files[:51]
            self._size = 51
        elif phase == 'val':
            if self.off_files:
                self.off_files = self.off_files[51:]
            if self.evecs_files:
                self.evecs_files = self.evecs_files[51:]
            if self.shot_files:
                self.shot_files = self.shot_files[51:]
            if self.corr_files:
                self.corr_files = self.corr_files[51:]
            if self.dist_files:
                self.dist_files = self.dist_files[51:]
            self._size = 20

        # reset combinations
        self._combinations = list(product(range(self._size), repeat=2))
