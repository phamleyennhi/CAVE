import numpy as np
import itertools
import torch
from sklearn.metrics import confusion_matrix
from pytorch3d.renderer import camera_position_from_spherical_angles
from src.lib.MeshUtils import (
    MeshInterpolateModule,
    load_off
)
def display_results(class_gds, class_preds, pose_errors, thresholds=[np.pi / 6, np.pi / 18]):
    class_gds = np.array(class_gds)
    class_preds = np.array(class_preds)
    acc = np.mean(class_gds == class_preds)
    print("*** Results: ***\nClassification accuracy : {acc:.03f}".format(acc=acc))
    
    for threshold in thresholds:
        pose_acc = np.mean(np.array(pose_errors) < threshold)
        print("Pose accuracy (error < {threshold:.03f}): {acc:.03f}".format(threshold=threshold, acc=pose_acc))
    
    pose_err_median = 180 / np.pi * np.median(np.array(pose_errors))
    print("Pose median error: {med_err:.03f}°".format(med_err=pose_err_median))
    

    # print confusion matrix
    print("Confusion matrix:\n", (confusion_matrix(class_gds, class_preds)))


def get_object_texture(config, objects_texture, class_, nb_vertices_list) -> torch.Tensor:
    if isinstance(class_, str):
        class_ = config.classes.index(class_)
    max_n = max(nb_vertices_list)
    return objects_texture[class_ * max_n : class_ * max_n + nb_vertices_list[class_]]


def prepare_rendering_for_init(config, objects_texture, n_list_set, rasterizer):
    ## Pre-rendering for all classes
    azum_s = np.linspace(0, np.pi * 2, 12, endpoint=False)
    elev_s = np.linspace(-np.pi / 6, np.pi / 3, 4)
    theta_s = np.linspace(-np.pi / 6, np.pi / 6, 3)
    get_samples = list(itertools.product(azum_s, elev_s, theta_s))
    pre_rendered_maps = []
    for c, class_ in enumerate(config.dataset.classes):
        xvert, xface = load_off(config.mesh_path % class_, to_torch=True)
        object_texture = get_object_texture(config, objects_texture, c, n_list_set)
        inter_module = MeshInterpolateModule(
            xvert,
            xface,
            object_texture.cuda(),
            rasterizer,
            # post_process=center_crop_fun(map_shape, (render_image_size,) * 2),
        )
        object_texture = object_texture.cpu()
        inter_module = inter_module.cuda()
        lookup_maps = []
        for sample_ in get_samples:
            t = torch.ones(1).cuda() * sample_[2]
            cam_pos = camera_position_from_spherical_angles(
                config.dataset.distance,
                sample_[1],
                sample_[0],
                degrees=False,
            ).cuda()
            projected_map = inter_module(cam_pos, t).squeeze()
            lookup_maps.append(projected_map)
        pre_rendered_maps.append(torch.stack(lookup_maps).cuda())
        inter_module = None
    return get_samples, pre_rendered_maps