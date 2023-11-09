""" PointNet2 backbone for feature learning.
    Author: Charles R. Qi
"""
import os
import sys
import torch
import torch.nn as nn
from ipdb import set_trace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModule, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, feature_dim_coff=1):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        self.sa1 = PointnetSAModule(
                npoint=512,
                radius=0.04,
                nsample=32,
                mlp=[input_feature_dim, 64*feature_dim_coff, 64*feature_dim_coff, 128*feature_dim_coff],
                use_xyz=True,
            )

        self.sa2 = PointnetSAModule(
                npoint=256,
                radius=0.1,
                nsample=16,
                mlp=[128*feature_dim_coff, 128*feature_dim_coff, 128*feature_dim_coff, 256*feature_dim_coff],
                use_xyz=True,
            )

        self.sa3 = PointnetSAModule(
                npoint=None,
                radius=None,
                nsample=None,
                mlp=[256*feature_dim_coff, 256*feature_dim_coff, 512*feature_dim_coff, 1024*feature_dim_coff],
                use_xyz=True,
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,D,K)
                XXX_inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        xyz, features = self._break_up_pc(pointcloud)

        # --------- 3 SET ABSTRACTION LAYERS ---------
        xyz, features = self.sa1(xyz, features)

        xyz, features = self.sa2(xyz, features) 

        xyz, features = self.sa3(xyz, features) 

        return features, xyz