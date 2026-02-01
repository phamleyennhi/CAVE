from typing import List
import warnings
import os
from pathlib import Path

import BboxTools as bbt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

class Pascal3DPlus(Dataset):
    def __init__(self, config, occlusion, transforms, max_n, test=False, enable_cache=True):
        
        self.for_test = test
        self.max_n = max_n
        self.transforms = transforms

        self.occlusion = occlusion
        assert self.occlusion in config.occlusion_levels, f"Invalid occlusion level, must be one of {config.occlusion_levels}"
        
        if self.for_test:
            root_path = Path(
                config.paths.root, 
                config.paths.eval_ood if self.occlusion else config.paths.eval_iid,
            )
        else:
            root_path = Path(
                config.paths.root,
                config.paths.training
            )

        self.weighted = config.weighted

        self.image_path = Path(root_path, config.paths.imgs)
        self.annotation_path = Path(root_path, config.paths.annot)
        list_path = Path(root_path, config.paths.img_list)

        self.classes: List = config.classes
        self.label_list = []
        self.file_list = []
        if self.occlusion == "":
            for class_ in self.classes: 
                print(class_)
                deeper_dir = list_path / class_ 
                print(deeper_dir)
                if not deeper_dir.exists():
                    warnings.warn(f"Class {class_} not found in {list_path}, skipping it.")
                    continue
                self.file_list = [
                    os.path.join(class_, l.strip())
                    for l in open(deeper_dir / "mesh01.txt").readlines()
                ] + self.file_list
            for f_dir in self.file_list:
                img_class = f_dir.split("/")[0]
                label = self.classes.index(img_class)
                self.label_list.append(label)
        else:
            for class_ in self.classes:
                deeper_dir = list_path / f"{class_}{occlusion}"
                self.file_list = [
                    os.path.join(f"{class_}{occlusion}", l.strip())
                    for l in open(deeper_dir / "mesh01.txt").readlines()
                ] + self.file_list
            for f_dir in self.file_list:
                img_class = f_dir.split("/")[0].strip(self.occlusion)
                label = self.classes.index(img_class)
                self.label_list.append(label)

        self.enable_cache = enable_cache
        self.cache_img = dict()
        self.cache_anno = dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name_img = self.file_list[item]
        this_name = name_img.split("/")[-1].split(".")[0]

        if name_img in self.cache_anno.keys():
            annotation_file = self.cache_anno[name_img]
            img = self.cache_img[name_img]
        else:
            if not name_img.endswith(".JPEG"):
                name_img += ".JPEG"
            img = Image.open(self.image_path / name_img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            annotation_file = np.load(self.annotation_path / (name_img.split(".")[0] + ".npz"), allow_pickle=True)

            if self.enable_cache:
                self.cache_anno[name_img] = dict(annotation_file)
                self.cache_img[name_img] = img

        box_obj = bbt.from_numpy(annotation_file["box_obj"])
        obj_mask = np.zeros(box_obj.boundary, dtype=np.float32)
        box_obj.assign(obj_mask, 1)

        kp = annotation_file["cropped_kp_list"]
        iskpvisible = annotation_file["visible"] == 1

        if self.weighted:
            iskpvisible = iskpvisible * annotation_file["kp_weights"]

        if not self.for_test:
            iskpvisible = np.logical_and(
                iskpvisible,
                np.all(kp >= np.zeros_like(kp), axis=1),
            )
            iskpvisible = np.logical_and(
                iskpvisible,
                np.all(kp < np.array([img.size[::-1]]), axis=1),
            )

        kp = np.max([np.zeros_like(kp), kp], axis=0)
        kp = np.min([np.ones_like(kp) * (np.array([img.size[::-1]]) - 1), kp], axis=0)

        pose_ = np.array(
            [
                annotation_file["distance"],
                annotation_file["elevation"],
                annotation_file["azimuth"],
                annotation_file["theta"],
            ],
            dtype=np.float32,
        )
        label = self.label_list[item]
        padded_dimension = self.max_n - kp.shape[0]
        kp = np.pad(
            kp,
            pad_width=((0, padded_dimension), (0, 0)),
            mode="constant",
            constant_values=-1,
        )
        iskpvisible = np.pad(
            iskpvisible,
            pad_width=(0, padded_dimension),
            mode="constant",
            constant_values=False,
        )
        index = np.array([self.max_n * label + k for k in range(self.max_n)])
        print(annotation_file['occ_mask'].shape)
        sample = {
            "img": img,
            "kp": kp,
            "iskpvisible": iskpvisible,
            "this_name": this_name,
            "obj_mask": obj_mask,
            "box_obj": box_obj.shape,
            "box": np.array(box_obj.bbox).reshape(-1),
            "pose": pose_,
            "label": label,
            "y_idx": index,
            "name_img": name_img,
            "principal": annotation_file["principal"],
            "focal": annotation_file["focal"],
            "cad_index": annotation_file["cad_index"],
            "height": annotation_file["height"],
            "width": annotation_file["width"],
            "cropped_occ_mask": np.array(annotation_file['cropped_occ_mask'], dtype=np.float32),
            "occ_mask": np.array(annotation_file['occ_mask'], dtype=np.float32),
            "padding": np.array(annotation_file['padding'], dtype=np.float32)
        }

        if self.transforms:
            sample = self.transforms(sample)
        return sample


class ToTensor:
    def __init__(self):
        self.trans = transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "iskpvisible" in sample and type(sample["iskpvisible"]) is not torch.Tensor:
            sample["iskpvisible"] = torch.Tensor(sample["iskpvisible"])
        if "kp" in sample and type(sample["kp"]) is not torch.Tensor:
            sample["kp"] = torch.Tensor(sample["kp"])
        return sample


class Normalize:
    def __init__(self):
        self.trans = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])

        return sample


def prepare_dataloader(config, max_n, use_test=True, occlusion="", shuffle=True):
    # prepare dataset
    Pascal3D_dataset = Pascal3DPlus(
        config=config.dataset,
        transforms=transforms.Compose([ToTensor(), Normalize()]),
        max_n=max_n,
        occlusion=occlusion,
        test=use_test,
    ) 
    # prepare dataloader
    Pascal3D_dataloader = DataLoader(
        Pascal3D_dataset,
        batch_size=config.inference.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
    )
    return Pascal3D_dataset, Pascal3D_dataloader

def prepare_dataloader_per_class(dataset: Dataset, shuffle=False, batch_size=1):
    subsets = {target: Subset(dataset, [i for i, label in enumerate(dataset.label_list) if label == target]) for target, _ in enumerate(dataset.classes)}
    loaders = {target: DataLoader(subset, shuffle=shuffle, batch_size=batch_size) for target, subset in subsets.items()}

    return loaders