from scipy.linalg import logm
import itertools
import numpy as np
import torch
from src.lib.ProcessCameraParameters import get_transformation_matrix
from pytorch3d.renderer import camera_position_from_spherical_angles

def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
    )


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5
    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]


def cal_err(gt, pred):
    # return radius
    return ((logm(np.dot(np.transpose(pred), gt)) ** 2).sum()) ** 0.5 / (2.0**0.5)


def loss_fun(obj_s: torch.Tensor, clu_s: torch.Tensor = None):
    if clu_s is None:
        return torch.ones(1, device=obj_s.device) - torch.mean(obj_s)
    return torch.ones(1, device=obj_s.device) - (
        torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s)
    )


def mask_out_loss_fun(
    obj_s: torch.Tensor,
    clu_s: torch.Tensor,
    object_height,
    object_width,
):
    obj_s = obj_s[
        object_height[0] : object_height[1],
        object_width[0] : object_width[1],
    ]
    clu_s = clu_s[
        object_height[0] : object_height[1],
        object_width[0] : object_width[1],
    ]
    if clu_s is None:
        return torch.ones(1, device=obj_s.device) - torch.mean(obj_s)
    return torch.ones(1, device=obj_s.device) - (
        torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s)
    )
    
def get_init_pos(
    samples,
    pre_rendered_maps,
    predicted_map,
    set_distance,
    clutter_score=None,
    device="cpu",
):
    get_loss = []
    for rendered_map in pre_rendered_maps:
        object_score = torch.sum(rendered_map * predicted_map, dim=0)
        loss = loss_fun(object_score, clutter_score)
        get_loss.append(loss.cpu().detach().numpy())
    sample_idx = np.argmin(get_loss)
    azum, elev, theta = samples[sample_idx]
    C = camera_position_from_spherical_angles(
        set_distance,
        elev,
        azum,
        degrees=False,
        device=device,
    )
    return C.detach(), torch.ones(1, device=device) * theta

