import os 
import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from src.evaluation import ATTRIBUTION_PATH, IMG_ATTRIBUTION_PATH

def compute_concept_dsc(method, img_name, cls_idx, cls_label, annot_tranform, num_concepts=20, dataset="PartPascal", eps=1e-8, percentile=0.90, seed=""):
    ground_truth_masks = torch.load(os.path.join(ATTRIBUTION_PATH, "ground_truth", annot_tranform, "PartPascal", f"{cls_idx}_{cls_label}", f"{img_name}.pt"))

    img_attributions_path = os.path.join(IMG_ATTRIBUTION_PATH, method, dataset, f"k_{num_concepts}", f"{cls_idx}_{cls_label}{seed}", f"{img_name}.pt")
    if not os.path.exists(img_attributions_path):
        print("not existed")
        return None
    img_attributions = torch.load(img_attributions_path)
    
    part_localisation = torch.zeros(img_attributions.shape[0])
    for c_id in range(img_attributions.shape[0]): # loop through the concepts
        concept_attribution = torch.tensor(img_attributions[c_id])
        concept_attribution = torch.clamp(concept_attribution, min=0)
        if concept_attribution.sum() < eps:
            part_localisation[c_id] = -1
            continue
        sigma = torch.quantile(concept_attribution, percentile)
        concept_attribution = torch.where(concept_attribution >= sigma, concept_attribution, 0)
 
        max_overlap = 0
        for _, part_mask in ground_truth_masks.items():
            part_mask = torch.tensor(part_mask)
            part_mask = F.max_pool2d(part_mask.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1) # dilate
            part_mask = part_mask.squeeze(0).squeeze(0)
            concept_mask = (concept_attribution > 0).float()
            weighted = (part_mask * concept_attribution).sum() / (concept_attribution.sum() + eps)
            overlap1 = (part_mask * concept_mask).sum()
            overlap2 = concept_mask.sum() * weighted
            overlap = (overlap1 + overlap2) / (concept_mask.sum() + part_mask.sum() + eps)
            max_overlap = max(overlap, max_overlap)
        part_localisation[c_id] = max_overlap
    
    return part_localisation


