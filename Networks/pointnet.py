# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# from torch.autograd import Variable
# import numpy as np
# import torch.nn.functional as F
# from ipdb import set_trace

# class STN3d(nn.Module):
#     def __init__(self, channel):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
#             batchsize, 1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x


# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k * k)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)

#         self.k = k

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
#             batchsize, 1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x

# # class PointNetEncoder(nn.Module):
# #     def __init__(self, global_feat=True, feature_transform=False, channel=3):
# #         super(PointNetEncoder, self).__init__()
# #         #self.stn = STN3d(channel)
# #         self.conv1 = torch.nn.Conv1d(channel, 4, 1)
# #         self.conv2 = torch.nn.Conv1d(4, 16, 1)
# #         self.conv3 = torch.nn.Conv1d(16, 64, 1)
# #         self.conv4 = torch.nn.Conv1d(64, 256, 1)
# #         self.conv5 = torch.nn.Conv1d(256, 1024, 1)

# #         self.bn1 = nn.BatchNorm1d(4)
# #         self.bn2 = nn.BatchNorm1d(16)
# #         self.bn3 = nn.BatchNorm1d(64)
# #         self.bn4 = nn.BatchNorm1d(256)
# #         self.bn5 = nn.BatchNorm1d(1024)

# #         self.global_feat = global_feat
# #         self.feature_transform = feature_transform
# #         if self.feature_transform:
# #             self.fstn = STNkd(k=64)

# #     def forward(self, x):
# #         B, D, N = x.size()
# #         #trans = self.stn(x)
# #         x = x.transpose(2, 1)  # [B, N, D]
# #         if D > 3 :
# #             x, feature = x.split(3,dim=2)
# #         #x = torch.bmm(x, trans)
# #         if D > 3:
# #             x = torch.cat([x,feature],dim=2)
# #         x = x.transpose(2, 1)
# #         x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

# #         if self.feature_transform:
# #             trans_feat = self.fstn(x)
# #             x = x.transpose(2, 1)
# #             x = torch.bmm(x, trans_feat)
# #             x = x.transpose(2, 1)
# #         else:
# #             trans_feat = None

# #         pointfeat = x
# #         x = F.relu(self.bn2(self.conv2(x)))
# #         x = F.relu(self.bn3(self.conv3(x)))
# #         x = F.relu(self.bn4(self.conv4(x)))
# #         x = self.bn5(self.conv5(x))
# #         x = torch.max(x, 2, keepdim=True)[0]
# #         x = x.view(-1, 1024)  # global feature: [B, 1024]
# #         if self.global_feat:
# #             return x, None, trans_feat
# #             #return x, trans, trans_feat
# #         else:
# #             x = x.view(-1, 1024, 1).repeat(1, 1, N)  # N  [B, 1024, N]
# #             return torch.cat([x, pointfeat], 1), None, trans_feat
# #             #return torch.cat([x, pointfeat], 1), trans, trans_feat

# class PointNetEncoder(nn.Module):
#     def __init__(self, global_feat=True, feature_transform=False, channel=3):
#         super(PointNetEncoder, self).__init__()
#         #self.stn = STN3d(channel)
#         self.conv1 = torch.nn.Conv1d(channel, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)

#     def forward(self, x):
#         B, D, N = x.size()
#         #trans = self.stn(x)
#         x = x.transpose(2, 1)  # [B, N, D]
#         if D > 3 :
#             x, feature = x.split(3,dim=2)
#         #x = torch.bmm(x, trans)
#         if D > 3:
#             x = torch.cat([x,feature],dim=2)
#         x = x.transpose(2, 1)
#         x = F.relu(self.conv1(x))  # [B, 64, N]

#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2, 1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2, 1)
#         else:
#             trans_feat = None

#         pointfeat = x
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)  # global feature: [B, 1024]
#         if self.global_feat:
#             return x, None, trans_feat
#             #return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, N)  # N  [B, 1024, N]
#             return torch.cat([x, pointfeat], 1), None, trans_feat
#             #return torch.cat([x, pointfeat], 1), trans, trans_feat


# def feature_transform_reguliarzer(trans):
#     d = trans.size()[1]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.cuda()
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
#     return loss

# """refer to https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py."""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from ipdb import set_trace
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# NOTE: removed BN
class PointNetEncoder(nn.Module):
    def __init__(self, num_points=1024, global_feat=True, in_dim=3, out_dim=1024, feature_transform=False, **args):
        super(PointNetEncoder, self).__init__()
        self.num_points = num_points
        self.out_dim = out_dim
        self.feature_transform = feature_transform
        # self.stn = STN3d(in_dim=in_dim)
        self.stn = STNkd(k=in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, out_dim, 1)
        self.global_feat = global_feat
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, **args):
        n_pts = x.shape[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        pointfeat = x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)
        if self.global_feat:
            return x, 0, 0
        else:
            x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print("stn", out.size())
    print("loss", feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print("stn64d", out.size())
    print("loss", feature_transform_regularizer(out))

    pointfeat_g = PointNetEncoder(global_feat=True, num_points=2500)
    out = pointfeat_g(sim_data)
    print("global feat", out.size())

    pointfeat = PointNetEncoder(global_feat=False, num_points=2500)
    out = pointfeat(sim_data)
    print("point feat", out.size())