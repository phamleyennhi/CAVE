import os
import torch
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import camera_position_from_spherical_angles, RasterizationSettings, PerspectiveCameras, MeshRasterizer

import src.dataset.Pascal3DPlus as p3d_imagenet

from src.lib.config import load_config, parse_args
from src.lib.get_n_list import compute_max_n
from src.lib.MeshUtils import load_off, campos_to_R_T, pre_process_mesh_pascal

MESH_PATHS = {
    "P3DImageNet": "PATH_TO_P3DIMAGENET_CAD",
    "OOD_CV": "PATH_TO_OOD_CV_CAD",
    "ImageNet3D": "PATH_TO_IMAGENET3D_CAD"
}

def pixel_to_face(cls_label, sample, rasterizer, dataset_name, device='cuda'):
    MESH_PATH = MESH_PATHS[dataset_name]

    cad_idx = sample["cad_index"][0].item()

    xverts, xfaces = load_off(os.path.join(MESH_PATH, f"{cls_label}/0{cad_idx}.off"), to_torch=True)
    xverts = pre_process_mesh_pascal(xverts)
    mesh = Meshes(verts=[xverts], faces=[xfaces], textures=None)
    mesh = mesh.to(device)

    _, elev, azum, theta = tuple(sample["pose"][0])
    theta = torch.ones(1, device=device) * theta
    C = camera_position_from_spherical_angles(5, elev, azum, degrees=False, device=device)
    R, T = campos_to_R_T(C, theta, device=device)

    fragments = rasterizer(mesh, R=R, T=T)
    pix_to_face = fragments.pix_to_face.squeeze().cpu().numpy()

    if occlusion != "":
        occ_mask = sample["cropped_occ_mask"].squeeze().numpy()
        occ_mask = (occ_mask == 0)
        pix_to_face *= occ_mask
        pix_to_face[pix_to_face == 0] = -1
    return torch.from_numpy(pix_to_face).cuda()



if __name__ == "__main__":
    args = parse_args()
    config = load_config(args, load_default_config=False, log_info=False)

    # compute no. gaussians list
    max_n, n_list_set = compute_max_n(config)
    num_classes = len(config.dataset.classes)
    down_sample_rate = config.model.down_sample_rate
    occlusion = args.occlusion

    # cls_idx
    cls_idx = args.cls_idx
    cls_label = config.dataset.classes[cls_idx]

    dataset_name = args.dataset

    test_dataset, _ = p3d_imagenet.prepare_dataloader(config, max_n, use_test=True, occlusion=occlusion)
    loaders = p3d_imagenet.prepare_dataloader_per_class(test_dataset)

    cls_loader = loaders[cls_idx]

    # rasterizer
    down_sample_rate = 1
    set_distance = 5.0
    classification_size = (640, 800) 
    render_image_size = max(classification_size) // down_sample_rate
    map_shape = (
        classification_size[0] // down_sample_rate,
        classification_size[1] // down_sample_rate,
    )
    cameras = PerspectiveCameras(
        focal_length=3000 // down_sample_rate,
        principal_point=((map_shape[1] // 2, map_shape[0] // 2),),
        image_size=(map_shape,),
        in_ndc=False,
    ).cuda()
    raster_settings = RasterizationSettings(
        image_size=map_shape,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    for j, sample in enumerate(tqdm(cls_loader)):
        img_name = sample["name_img"][0].split("/")[1].split(".")[0]
        fragments = pixel_to_face(cls_label, sample, rasterizer, dataset_name)
        torch.save(fragments, os.path.join("PATH_TO_PIX_TO_FACE", f"{img_name}.pt"))
