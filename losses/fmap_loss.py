# The computation of SURFMNetLoss is adapted from https://github.com/pvnieo/SURFMNet-pytorch

import numpy as np
import torch
import torch.nn as nn
from utils.registry import LOSS_REGISTRY
from .loss_util import FrobeniusLoss


@LOSS_REGISTRY.register()
class AlignLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        assert loss_weight >= 0, f'Invalid loss weight: {loss_weight}'
        super(AlignLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, Cyx, Pxy, evecs_x, evecs_y):
        """
        Forward pass
        Args:
            Cyx: functional map shape y -> shape x. [B, K, K].
            Pxy: permutation matrix shape x -> shape y. [B, Vx, Vy].
            evecs_x: eigenvectors of Laplacian of shape x. [B, Vx, K].
            evecs_y: eigenvectors of Laplacian of shape y. [B, Vy, K].

        Returns:
            loss
        """
        criterion = FrobeniusLoss()
        align_loss = criterion(torch.bmm(evecs_x, Cyx), torch.bmm(Pxy, evecs_y))
        return self.loss_weight * align_loss


@LOSS_REGISTRY.register()
class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity + Descriptor Preservation
    """

    def __init__(self, w_bij=1e3, w_orth=1e3, w_lap=1.0, w_pre=1e5, sub_pre=0.2):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
            w_pre (float, optional): Descriptor preservation via commutativity penalty weight. Defaults to 1e5.
            sub_pre (float, optional): Percentage of subsampled vertices used to compute
                                    descriptor preservation via commutativity penalty. Defaults to 0.2.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0 and w_pre >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap
        self.w_pre = w_pre
        self.sub_pre = sub_pre

    def forward(self, C12, C21, feat_1, feat_2, evecs_1, evecs_2, evecs_trans1, evecs_trans2, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            feat_1 (torch.Tensor): learned feature vectors of shape 1. Shape: [N, V, C]
            feat_2 (torch.Tensor): learned feature vectors of shape 2. Shape: [N, V, C]
            evecs_1 (torch.Tensor): eigenvectors of shape 1. Shape: [N, V, K].
            evecs_2 (torch.Tensor): eigenvectors of shape 2. Shape: [N, V, K].
            evecs_trans1 (torch.Tensor): pseudo inverse eigenvectors of shape 1. Shape: [N, K, V].
            evecs_trans2 (toexh.Tensor): pseudo inverse eigenvectors of shape 2. Shape: [N, K, V].
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = FrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)

        # Bijectivity penalty
        if self.w_bij > 0:
            bijectivity_loss = criterion(torch.bmm(C12, C21), eye_batch) + criterion(torch.bmm(C21, C12), eye_batch)
            bijectivity_loss *= self.w_bij
        else:
            bijectivity_loss = 0.0

        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C12.transpose(1, 2), C12), eye_batch) + \
                                 criterion(torch.bmm(C21.transpose(1, 2), C21), eye_batch)
            orthogonality_loss *= self.w_orth
        else:
            orthogonality_loss = 0.0

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            laplacian_loss = criterion(torch.einsum('abc,ac->abc', C12, evals_1),
                                       torch.einsum('ab,abc->abc', evals_2, C12))
            laplacian_loss += criterion(torch.einsum('abc,ac->abc', C21, evals_2),
                                        torch.einsum('ab,abc->abc', evals_1, C21))
            laplacian_loss *= self.w_lap
        else:
            laplacian_loss = 0.0

        # Descriptor preservation via commutativity
        if self.w_pre > 0:
            num_desc = int(feat_1.shape[2] * self.sub_pre)
            desc_ind = np.random.choice(feat_1.shape[2], num_desc, replace=False)
            feat_1 = feat_1[:, :, desc_ind].transpose(1, 2).unsqueeze(2)
            feat_2 = feat_2[:, :, desc_ind].transpose(1, 2).unsqueeze(2)
            M_1 = torch.einsum('abcd,ade->abcde', feat_1, evecs_1)
            M_1 = torch.einsum('afd,abcde->abcfe', evecs_trans1, M_1)
            M_2 = torch.einsum('abcd,ade->abcde', feat_2, evecs_2)
            M_2 = torch.einsum('afd,abcde->abcfe', evecs_trans2, M_2)
            C12_expand = torch.repeat_interleave(C12.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1)
            C21_expand = torch.repeat_interleave(C21.unsqueeze(1).unsqueeze(1), repeats=num_desc, dim=1)
            source1, target1 = torch.einsum('abcde,abcef->abcdf', C12_expand, M_1), torch.einsum('abcef,abcfd->abced',
                                                                                                 M_2, C12_expand)
            source2, target2 = torch.einsum('abcde,abcef->abcdf', C21_expand, M_2), torch.einsum('abcef,abcfd->abced',
                                                                                                 M_1, C21_expand)
            preservation_loss = criterion(source1, target1) + criterion(source2, target2)
            preservation_loss *= self.w_pre
        else:
            preservation_loss = 0.0

        return {'l_bij': bijectivity_loss, 'l_orth': orthogonality_loss, 'l_lap': laplacian_loss,
                'l_pre': preservation_loss}
