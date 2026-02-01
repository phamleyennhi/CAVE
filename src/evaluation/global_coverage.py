import os 
import torch
import torch.nn.functional as F

from tqdm import tqdm
from src.evaluation import ATTRIBUTION_PATH, IMG_ATTRIBUTION_PATH, to_torch_tensor


def compute_concept_all_parts_coverage(method, img_name, cls_idx, cls_label, annot_transform, num_concepts=20, dataset="PartPascal", eps=1e-8, percentile=0.90):
    
    ground_truth_masks = torch.load(os.path.join(ATTRIBUTION_PATH, "ground_truth", annot_transform, dataset, f"{cls_idx}_{cls_label}", f"{img_name}.pt"))
    
    img_attributions_path = os.path.join(IMG_ATTRIBUTION_PATH, method, dataset, f"k_{num_concepts}", f"{cls_idx}_{cls_label}", f"{img_name}.pt")
    if not os.path.exists(img_attributions_path):
        print("not existed")
        return -1

    # dilate ground_truth mask
    dilated_ground_truth_masks = {}
    for part_name, part_mask in ground_truth_masks.items():
        part_mask = torch.tensor(part_mask)
        part_mask = F.max_pool2d(part_mask.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1) # dilate
        part_mask = part_mask.squeeze(0).squeeze(0)
        dilated_ground_truth_masks[part_name] = part_mask

    # threshold & filter out invalid attributions
    orig_img_attributions = torch.load(img_attributions_path)
    orig_img_attributions = to_torch_tensor(orig_img_attributions)
    orig_img_attributions = torch.clamp(orig_img_attributions, min=0)
    if torch.isnan(orig_img_attributions).sum() > 0 or orig_img_attributions.sum() < eps:
        return -1

    img_attributions = []
    for c_id in range(orig_img_attributions.shape[0]): # loop through the concepts
        concept_attribution = torch.tensor(orig_img_attributions[c_id])
        sigma = torch.quantile(concept_attribution, percentile)
        concept_attribution = torch.where(concept_attribution >= sigma, concept_attribution, 0)
        img_attributions.append(concept_attribution)
    
    img_attributions = torch.stack(img_attributions)

    # normalisation
    img_attributions = img_attributions / img_attributions.sum()
    
    # concept coverage
    concept_attributions = img_attributions.sum(dim=0)

    # ground-truth coverage
    ground_truth_coverage = None
    for part_name, part_mask in dilated_ground_truth_masks.items():
        if ground_truth_coverage is None:
            ground_truth_coverage = part_mask
        else:
            ground_truth_coverage += part_mask
    ground_truth_coverage = torch.tensor(ground_truth_coverage)
    ground_truth_mask = (ground_truth_coverage > 0).float()
    concept_mask = (concept_attributions > 0).float()

    # compute global dsc 
    weighted = (concept_attributions * ground_truth_mask).sum() # ratio how much attributions fall into the ground-truth
    overlap1 = (ground_truth_mask * concept_mask).sum() # IoU
    overlap2 = concept_mask.sum() * weighted
    coverage = (overlap1 + overlap2) / (concept_mask.sum() + ground_truth_mask.sum())

    return coverage

