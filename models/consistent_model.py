import torch

from .base_model import BaseModel
from utils.logger import get_root_logger
from utils.registry import MODEL_REGISTRY
from utils.geometry_util import torch2np
from utils.sinkhorn_util import gumbel_sinkhorn
from utils.tensor_util import to_device


@MODEL_REGISTRY.register()
class ConsistentModel(BaseModel):
    """
    Consistent Model: Multi-Shape Matching with Cycle Consistency
    """

    def __init__(self, opt):
        super(ConsistentModel, self).__init__(opt)
        # hyper-parameters for Sinkhorn normalization
        self.temp = opt.get('temp', 0.2)
        self.n_iter = opt.get('n_iter', 10)
        if self.is_train:
            # hyper-parameters for functional map computation
            self.n_eig = opt.get('n_eig', 80)
            # hyper-parameters for changing n_eig during training
            self.milestones = opt.get('milestones')
            self.interval = opt.get('interval', 4000)
            self.restart = 0

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # predict
        feat_x, feat_y, P = self.predict(data_x, data_y)

        # compute functional map
        evecs_trans_x = data_x['evecs_trans'][:, :self.n_eig]
        evecs_trans_y = data_y['evecs_trans'][:, :self.n_eig]

        C_12, C_21 = self.networks['fmap_net'](feat_x, feat_y, evecs_trans_x, evecs_trans_y)

        # compute surfmnet loss
        evecs_x, evals_x = data_x['evecs'][:, :, :self.n_eig], data_x['evals'][:, :self.n_eig]
        evecs_y, evals_y = data_y['evecs'][:, :, :self.n_eig], data_y['evals'][:, :self.n_eig]
        self.loss_metrics = self.losses['surfmnet_loss'](C_12, C_21, feat_x, feat_y,
                                                         evecs_x, evecs_y,
                                                         evecs_trans_x, evecs_trans_y,
                                                         evals_x, evals_y)

        # compute the alignment loss |evecs_x*Cyx - Pxy*evecs_y|
        if self.restart >= self.interval:
            self.loss_metrics['l_fro'] = self.losses['align_loss'](C_21, P, evecs_x, evecs_y)
        else:
            self.loss_metrics['l_fro'] = self.losses['align_loss'](C_21.detach(), P, evecs_x, evecs_y)

    def optimize_parameters(self):
        # compute total loss
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k.startswith('l_'):
                loss += v

        # update loss metrics
        self.loss_metrics.update({'l_total': loss})

        # zero grad
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

        # backward pass
        loss.backward()

        # clip gradient for stability
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)

        # update weight
        for name in self.optimizers:
            self.optimizers[name].step()

    def update_model_per_iteration(self):
        super(ConsistentModel, self).update_model_per_iteration()
        self.restart += 1
        if self.milestones and self.curr_iter in self.milestones:
            old_n_eig = self.n_eig
            new_n_eig = self.milestones[self.curr_iter]

            logger = get_root_logger()
            logger.info(f'Update number of eigenfunctions from {old_n_eig} to {new_n_eig}')
            self.n_eig = new_n_eig
            self.restart = 0

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # start record
        timer.start()

        P = self.predict(data_x, data_y)[-1]

        # convert soft correspondence to p2p (shape y -> shape x)
        p2p = torch2np(P.argmax(dim=1).squeeze())  # [Vy]

        # finish record
        timer.record()

        # get geodesic distance matrix
        dist_x = data_x['dist']

        # get gt correspondence
        corr_x = data_x['sample']
        corr_y = data_y['sample']

        # convert torch.Tensor to numpy.ndarray
        dist_x = torch2np(dist_x.squeeze())
        corr_x = torch2np(corr_x.squeeze())
        corr_y = torch2np(corr_y.squeeze())

        # compute geodesic error
        geo_err = self.metrics['geo_error'](dist_x, corr_x, corr_y, p2p, return_mean=False)
        return geo_err, p2p

    def predict(self, data_x, data_y):
        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'], data_x['shot'])
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'], data_y['shot'])

        # make prediction
        pred_x = gumbel_sinkhorn(self.networks['classifier'](data_x['verts'], data_x['faces'], feat_x),
                                 temp=self.temp, n_iter=self.n_iter)
        pred_y = gumbel_sinkhorn(self.networks['classifier'](data_y['verts'], data_y['faces'], feat_y),
                                 temp=self.temp, n_iter=self.n_iter)

        # soft correspondence
        P = torch.bmm(pred_x, pred_y.transpose(1, 2))  # [B, Vx, Vy]

        return feat_x, feat_y, P
