import numpy as np
from ipdb import set_trace
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Networks.pointnet2.pointnet2_backbone import Pointnet2Backbone
from Networks.pointnet import PointNetEncoder
# from Networks.MHCA import MHCA

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class CondScoreModel(nn.Module):
    def __init__(self, marginal_prob_func, hidden_dim, embed_dim, obj_dim=6, class_num=10, state_dim=1, size_dim=2,
                 mode='target', relative=False, pointnet_network_type='new_3', pointnet_version='pt2', n_blocks=0, feature_dim_coff=1, space='euler'):
        super(CondScoreModel, self).__init__()
        self.marginal_prob_func = marginal_prob_func
        self.point_feat_dim = 1088
        hidden_dim = hidden_dim
        embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.mode = mode
        self.pointnet_version = pointnet_version
        if relative:
            hand_state_dim = 18
            if space == 'riemann':
                hand_state_dim = 18+18
        else:
            hand_state_dim = 25
            if space == 'riemann':
                hand_state_dim = 25+18
            
        self.n_blocks = n_blocks
        self.hand_global_enc = nn.Sequential(
            nn.Linear(hand_state_dim, hidden_dim),
            nn.ReLU(False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(False),
        )
        # obj pcl feature encoder
        if pointnet_version == 'pt':
            self.obj_enc = PointNetEncoder(global_feat=True, feature_transform=False, channel=3) # for pointnet
        elif pointnet_version == 'pt2':
            self.obj_enc = Pointnet2Backbone(feature_dim_coff=feature_dim_coff) # for pointnet2
        # self.obj_enc = PointNetEncoder() # for pointnet2
        # self.obj_cat_embed = nn.Embedding(301,512)

        if self.n_blocks < 1:
            self.obj_global_enc = nn.Sequential(
                    nn.Linear(1024, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim),
                    nn.ReLU(),
                )
        self.embed_sigma = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                        nn.Linear(embed_dim, embed_dim))

        if n_blocks < 1:
            self.init_enc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.point_feat_dim),
                nn.ReLU(),
            )

            # cond_dim = hidden_dim*2 + embed_dim*2 # consider wall
            if self.mode == 'target':
                cond_dim = embed_dim
            
            # self.mhca = MHCA(num_heads=2, inp_dim=self.point_feat_dim, hid_dim=self.point_feat_dim)
            ''' main backbone '''
            # # mlp1
            self.mlp1_main = nn.Sequential(
                nn.Linear((hidden_dim + embed_dim*2), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # # mlp2
            self.mlp2_main = nn.Sequential(
                nn.Linear(hidden_dim + embed_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hand_state_dim),
            )
        else:
            self.pre_dense_cond = nn.Linear(1024*feature_dim_coff, hidden_dim)
            self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
            # self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)

            for idx in range(n_blocks):
                setattr(self, f'b{idx+1}_dense1', nn.Linear(hidden_dim, hidden_dim))
                setattr(self, f'b{idx+1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
                setattr(self, f'b{idx+1}_dense1_cond', nn.Linear(hidden_dim, hidden_dim))
                # setattr(self, f'b{idx+1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

                setattr(self, f'b{idx+1}_dense2', nn.Linear(hidden_dim, hidden_dim))
                setattr(self, f'b{idx+1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
                setattr(self, f'b{idx+1}_dense2_cond', nn.Linear(hidden_dim, hidden_dim))
                # setattr(self, f'b{idx+1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

            self.act = nn.ReLU(False)
            self.post_dense = nn.Linear(hidden_dim, hand_state_dim)        

    def forward(self, batches, t, obj_feature=False):
        """
        batches = hand_batch, obj_batch
        hand_batch: [bs, 25, 1]
        obj_batch: [bs, 3, 1024]
        t: [bs] !! not [bs, 1] !!
        """
        hand_batch, obj_batch = batches
        batch_size = hand_batch.size(0)
        hand_dof = hand_batch.size(1)
        ''' get cond feat'''

        # sigma_feat: [num_nodes, embed_dim]
        sigma_feat = F.relu(self.embed_sigma(t.squeeze(-1)),inplace=False)

        # total_cond_feat: [num_nodes, hidden_dim*2+embed_dim*2]
        # obj_feat,_, _ = self.obj_enc(obj_batch.reshape(batch_size,-1,3)) # B x 1024

        ## no cuda pointnet2
        # obj_feat,_ = self.obj_enc(obj_batch) # B x 1024
        # obj_feat = self.obj_global_enc(obj_feat)
        if self.pointnet_version == 'pt':
            obj_feat,_,_ = self.obj_enc(obj_batch) # B x 1024
        elif self.pointnet_version == 'pt2':
            ## cuda pointnet2
            obj_feat,_ = self.obj_enc(obj_batch.reshape(batch_size,-1,3)) # B x 1024
        ## pointnet

        if obj_feature:
            obj_feat_fr = obj_feat.clone()

        if self.n_blocks < 1:
            ''' get init x feat '''
            hand_global_feat = self.hand_global_enc(hand_batch.reshape(batch_size,-1))
            obj_feat = self.obj_global_enc(obj_feat.reshape(batch_size,-1))

            # obj_feat = torch.arange(0,batch_size,device=hand_batch.device)
            # obj_feat = self.obj_cat_embed(obj_feat)
            if self.mode == 'target':
                total_cond_feat = torch.cat([sigma_feat, obj_feat], dim=-1)  #
                # total_cond_feat = sigma_feat

            ''' main backbone of x '''
            x = torch.cat([hand_global_feat, total_cond_feat], -1)
            x = self.mlp1_main(x)
            x = torch.cat([x, total_cond_feat], -1)
            x = self.mlp2_main(x)
        else:
            obj_feat = obj_feat.reshape(batch_size,-1)
            obj_feat = self.pre_dense_cond(obj_feat)

            x = self.hand_global_enc(hand_batch.reshape(batch_size,-1))
            x = x + self.pre_dense_t(sigma_feat)
            x = x + obj_feat
            # x = self.pre_gnorm(x)
            x = self.act(x)
            
            for idx in range(self.n_blocks):
                x1 = getattr(self, f'b{idx+1}_dense1')(x)
                x1 = x1 + getattr(self, f'b{idx+1}_dense1_t')(sigma_feat)
                x1 = x1 + getattr(self, f'b{idx+1}_dense1_cond')(obj_feat)
                # x1 = getattr(self, f'b{idx+1}_gnorm1')(x1)
                x1 = self.act(x1)
                # dropout, maybe
                # x1 = self.dropout(x1)

                x2 = getattr(self, f'b{idx+1}_dense2')(x1)
                x2 = x2 + getattr(self, f'b{idx+1}_dense2_t')(sigma_feat)
                x2 = x2 + getattr(self, f'b{idx+1}_dense2_cond')(obj_feat)
                # x2 = getattr(self, f'b{idx+1}_gnorm2')(x2)
                x2 = self.act(x2)
                # dropout, maybe
                # x2 = self.dropout(x2)

                x = x + x2

            x = self.post_dense(x)
        # normalize the output
        
        _, std = self.marginal_prob_func(x, t) 
        x = x / (std + 1e-7)
        if obj_feature:
            return x, obj_feat_fr
        else:
            return x