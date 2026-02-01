import os 
import torch
import cv2
import numpy as np
from src.evaluation import IMG_ATTRIBUTION_PATH, to_torch_tensor

def rescale_bounding_boxes(original_shape: tuple, new_shape: tuple,
                           bounding_boxes: list) -> list:
    """
    Rescales bounding box coords according to new image size
    :param original_shape: (W, H)
    :param new_shape: (W, H)
    :param bounding_boxes: [x1, y1, x2, y2]
    :return: scaled bbox coords
    """
    original_w, original_h = original_shape
    new_w, new_h = new_shape
    bounding_boxes = np.array(bounding_boxes, dtype=np.float64)
    scale_h, scale_w = new_h / original_h, new_w / original_w
    bounding_boxes[0] *= scale_w
    bounding_boxes[1] *= scale_h
    bounding_boxes[2] *= scale_w
    bounding_boxes[3] *= scale_h
    bounding_boxes = np.clip(bounding_boxes, a_min=0, a_max=None)
    bounding_boxes = bounding_boxes.astype(np.uint32).tolist()
    return bounding_boxes


def compute_object_coverage(method, img_name, bbox, cls_idx, cls_label, num_concepts=20, dataset="PartPascal", eps=1e-8, percentile=0.90, seed=""):
    img_attributions_path = os.path.join(IMG_ATTRIBUTION_PATH, method, dataset, f"k_{num_concepts}", f"{cls_idx}_{cls_label}{seed}", f"{img_name}.pt")

    if not os.path.exists(img_attributions_path):
        print("not existed", img_attributions_path)
        return -1

    # thresholding
    orig_img_attributions = torch.load(img_attributions_path)
    orig_img_attributions = to_torch_tensor(orig_img_attributions)
    orig_img_attributions = torch.clamp(orig_img_attributions, min=0)

    img_attributions = []
    for c_id in range(orig_img_attributions.shape[0]): # loop through the concepts
        concept_attribution = torch.tensor(orig_img_attributions[c_id])
        sigma = torch.quantile(concept_attribution, percentile)
        concept_attribution = torch.where(concept_attribution >= sigma, concept_attribution, 0)
        img_attributions.append(concept_attribution)
    
    if len(img_attributions) == 0:
        return -1

    img_attributions = torch.stack(img_attributions)

    # normalisation
    img_attributions = img_attributions / (img_attributions.sum() + eps) 
    
    # concept coverage
    concept_attributions = img_attributions.sum(dim=0)
 
    # object coverage
    if method == "cave": 
        object_attribution = (concept_attributions * bbox)
        bbox_sum = bbox.sum()
    else:
        xmin, ymin, xmax, ymax = bbox
        bbox_sum = (xmax - xmin) * (ymax - ymin)
        object_attribution = concept_attributions[ymin:ymax, xmin:xmax]
    
    weighted = object_attribution.sum() / (concept_attributions.sum() + eps)
    concept_mask = torch.where(concept_attributions > 0, 1, 0)
    iou = torch.where(object_attribution > 0, 1, 0)

    overlap1 = (weighted * concept_mask).sum()
    overlap2 = iou.sum()
    return (overlap1 + overlap2) / (concept_mask.sum() + bbox_sum + eps)

