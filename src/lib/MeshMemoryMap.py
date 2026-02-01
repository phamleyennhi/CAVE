import os

import BboxTools as bbt
import numpy as np
import torch
from src.lib.CalculateOcc import cal_occ_one_image
from src.lib.MeshUtils import load_off
from src.lib.ProcessCameraParameters import CameraTransformer
from src.lib.ProcessCameraParameters import get_anno
from src.lib.ProcessCameraParameters import Projector3Dto2D


def normalization(value):
    return (value - value.min()) / (value.max() - value.min())


def box_include_2d(self_box, other):
    return np.logical_and(
        np.logical_and(
            self_box.bbox[0][0] <= other[:, 0],
            other[:, 0] < self_box.bbox[0][1],
        ),
        np.logical_and(
            self_box.bbox[1][0] <= other[:, 1],
            other[:, 1] < self_box.bbox[1][1],
        ),
    )


def filter_points_per_pixel(points_2d, if_visible):
    """Mark as not visible if there are more than one points in one pixel

    Args:
        points_2d (array): 2d coordinates of all points
        if_visible (array): boolean array of visible points

    Returns:
        array: visible and unique points
    """

    pixels = points_2d[if_visible, :]
    _, idx = np.unique(pixels, axis=0, return_index=True)
    idx = np.nonzero(if_visible)[0][idx]
    visible_ = np.zeros_like(if_visible)
    visible_[idx] = True
    return visible_ & if_visible


class MeshLoader:
    def __init__(self, path):
        if os.path.isdir(path):
            file_list = os.listdir(path)

            l = len(file_list)
            file_list = ["%02d.off" % (i + 1) for i in range(l)]
        else:
            file_list = [path]
            l = 1
        self.mesh_points_3d = []
        self.mesh_triangles = []

        for fname in file_list:
            points_3d, triangles = load_off(os.path.join(path, fname))
            self.mesh_points_3d.append(points_3d)
            self.mesh_triangles.append(triangles)

    def __getitem__(self, item):
        return self.mesh_points_3d[item], self.mesh_triangles[item]

    def __len__(self):
        return len(self.mesh_points_3d)


class MeshConverter:
    def __init__(self, path):
        self.loader = MeshLoader(path=path)

    def get_one_no_box(self, annos, return_distance=False):
        points_3d, triangles = self.loader[0]
        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)
        cam_3d = CameraTransformer(
            annos,
        ).get_camera_position()  #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        distance = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** 0.5
        distance_ = normalization(distance)
        h, w = get_anno(annos, "height", "width")
        map_size = (h, w)

        if_visible = cal_occ_one_image(
            points_2d=points_2d,
            distance=distance_,
            triangles=triangles,
            image_size=map_size,
        )

        # handle the case that points are out of boundary of the image
        points_2d = np.max([np.zeros_like(points_2d), points_2d], axis=0)
        points_2d = np.min(
            [np.ones_like(points_2d) * (np.array([map_size]) - 1), points_2d],
            axis=0,
        )

        if return_distance:
            return points_2d, if_visible, distance_

        return points_2d, if_visible

    def get_one(self, annos, return_distance=False):
        points_3d, triangles = self.loader[0]
        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)
        cam_3d = CameraTransformer(
            annos,
        ).get_camera_position()  #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        distance = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** 0.5
        distance_ = normalization(distance)
        h, w = get_anno(annos, "height", "width")
        map_size = (h, w)

        if_visible = cal_occ_one_image(
            points_2d=points_2d,
            distance=distance_,
            triangles=triangles,
            image_size=map_size,
        )
        box_ori = bbt.from_numpy(get_anno(annos, "box_ori"))
        box_cropped = bbt.from_numpy(get_anno(annos, "box_obj").astype(np.int32))
        box_cropped.set_boundary(
            get_anno(annos, "box_obj").astype(np.int32)[4::].tolist(),
        )

        if_visible = np.logical_and(if_visible, box_include_2d(box_ori, points_2d))

        projection_foo = bbt.projection_function_by_boxes(box_ori, box_cropped)

        pixels_2d = projection_foo(points_2d)

        # handle the case that points are out of boundary of the image
        pixels_2d = np.max([np.zeros_like(pixels_2d), pixels_2d], axis=0)
        pixels_2d = np.min(
            [
                np.ones_like(pixels_2d) * (np.array([box_cropped.boundary]) - 1),
                pixels_2d,
            ],
            axis=0,
        )

        if return_distance:
            return pixels_2d, if_visible, distance_

        return pixels_2d, if_visible

    def get_visible_points_from_view(
        self,
        azimuth,
        elevation,
        theta,
        height=640,
        width=800,
        distance=5.0,
        focal=1.0,
        viewport=3000,
        principal=None,
        cad_index=1,
        margin=0,
        box_projection=None,
        debug=False,
    ):
        if principal is None:
            principal = np.array([0.5 * width, 0.5 * height])
        annos = {
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance,
            "focal": focal,
            "theta": theta,
            "principal": principal,
            "viewport": viewport,
        }
        points_3d, triangles = self.loader[cad_index - 1]
        vertices_idx = np.arange(points_3d.shape[0])
        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        cam_3d = CameraTransformer(
            annos,
        ).get_camera_position()  #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        dist = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** 0.5
        dist_ = normalization(dist)
        map_size = (height, width)

        if_visible = cal_occ_one_image(
            points_2d=points_2d,
            distance=dist_,
            triangles=triangles,
            image_size=map_size,
        )
        if_visible = filter_points_per_pixel(points_2d, if_visible)
        visible_points_2d = points_2d[if_visible]

        if box_projection is None:
            image_box = bbt.from_numpy(np.array([0, height, 0, width, height, width]))
            if_visible = if_visible & box_include_2d(image_box, points_2d)
            if_visible = filter_points_per_pixel(points_2d, if_visible)
            return vertices_idx[if_visible], points_2d[if_visible]

        # compute the smallest bounding box with same ratio as image and centered at the center of the bounding box
        ratio = float(height) / width
        min_x = np.min(visible_points_2d[:, 0] - margin)
        max_x = np.max(visible_points_2d[:, 0] + margin)
        min_y = np.min(visible_points_2d[:, 1] - margin)
        max_y = np.max(visible_points_2d[:, 1] + margin)
        center_x = float(min_x + max_x) / 2
        center_y = float(min_y + max_y) / 2
        tight_height = max_y - min_y
        tight_width = max_x - min_x
        if float(tight_height) / tight_width > ratio:
            tight_width = tight_height / ratio
        else:
            tight_height = tight_width * ratio
        min_x = center_x - tight_width / 2
        max_x = center_x + tight_width / 2
        min_y = center_y - tight_height / 2
        max_y = center_y + tight_height / 2
        box_tight = bbt.from_numpy(
            np.array([min_y, max_y, min_x, max_x]).astype(np.int32),
        )

        projection_foo = bbt.projection_function_by_boxes(box_tight, box_projection)
        point_2d_resized = (
            projection_foo(torch.from_numpy(visible_points_2d).flip(1).float())
            .flip(1)
            .long()
            .numpy()
        )

        if debug:
            return (
                vertices_idx[if_visible],
                visible_points_2d,
                box_tight,
                point_2d_resized,
            )
        # return visible vertices index, pixel_coordinates
        return vertices_idx[if_visible], point_2d_resized
