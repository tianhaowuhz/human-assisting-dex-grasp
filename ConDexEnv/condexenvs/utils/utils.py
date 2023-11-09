# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# python

import numpy as np
import torch
import random
import os
import torch.nn.functional as F
import transforms3d

def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

# EOF

# for generate trajectory
def rodrigues_to_rotation(rot, theta):
    # r [B, 3]
    # theta [B, 1]
    device = rot.device
    rot = (rot.permute(1, 0) / torch.norm(rot, dim=1)).permute(1, 0)
    B = rot.shape[0]
    rot = rot.unsqueeze(2)
    rot_T = rot.permute(0, 2, 1)
    rx = rot[:, 0, 0]
    ry = rot[:, 1, 0]
    rz = rot[:, 2, 0]
    M = torch.zeros(B, 3, 3, device=device)
    for i in range(B):
        M[i] = torch.tensor([[0, -rz[i], ry[i]],
                             [rz[i], 0, -rx[i]],
                             [-ry[i], rx[i], 0]],device=device)
    R = torch.eye(3,device=device).expand(B, 3, 3)
    R = (torch.cos(theta) * torch.eye(3,device=device).expand(B, 3, 3).permute(1, 2, 0)).permute(2, 0, 1) +        \
        ((1 - torch.cos(theta)) * (rot @ rot_T).permute(1, 2, 0)).permute(2, 0, 1) +    \
        (torch.sin(theta) * M.permute(1, 2, 0)).permute(2, 0, 1)
    return R


def sample_from_circle(theta):
    device = theta.device
    B = theta.shape[0]
    z = 1
    r = torch.abs(torch.tan(theta) * z)
    theta = torch.rand(B, 1, device=device) * 2 * np.pi - np.pi
    x = torch.cos(theta)
    y = torch.sin(theta)
    len = r * torch.rand(B, 1, device=device)
    x = x * len
    y = y * len
    z = torch.ones(B, 1, device=device)
    sample_vector = torch.cat((x, y, z), dim=1)
    sample_vector = (sample_vector.permute(1, 0) / torch.norm(sample_vector, dim=1)).permute(1, 0)
    return sample_vector.unsqueeze(2)


def angle_from_vector(vec_1, vec_2):
    dot = torch.sum(torch.multiply(vec_1, vec_2), dim=1)
    vec_1_norm = torch.norm(vec_1, dim=1)
    vec_2_norm = torch.norm(vec_2, dim=1)
    theta = torch.arccos(dot/(vec_1_norm * vec_2_norm))
    return theta

# dexgrasp net table
def plane2pose(plane_parameters):
        r3 = plane_parameters[:3]
        r2 = np.zeros_like(r3)
        r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0) if r3[2] * r3[2] <= 0.5 else (-r3[2], 0, r3[0])
        r1 = np.cross(r2, r3)
        pose = np.zeros([4, 4], dtype=np.float32)
        pose[0, :3] = r1
        pose[1, :3] = r2
        pose[2, :3] = r3
        pose[2, 3] = plane_parameters[3]
        pose[3, 3] = 1
        return pose
    
def plane2euler(plane_parameters, axes='sxyz'):
        pose = plane2pose(plane_parameters)
        T, R, Z, S = transforms3d.affines.decompose(pose)
        euler = transforms3d.euler.mat2euler(R, axes=axes)
        return T, euler
#####################################################################
###=====================pointcloud process========================###
#####################################################################
def farthest_point_sample(xyz, npoint, device, init=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if init is not None:
        farthest = torch.tensor(init).long().reshape(B).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx, device):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.size()[0]
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
