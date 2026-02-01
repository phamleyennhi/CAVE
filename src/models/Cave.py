import os
import torch
import torch.nn as nn
import numpy as np
import BboxTools as bbt
from src.models.KeypointRepresentationNet import NetE2E

class FeatureMatchingCAVE(nn.Module):
    def __init__(self, compare_bank, meta_gaussian_class_indices, class_labels, down_sample_rate=1):
        super().__init__()
        self.class_labels = class_labels
        self.down_sample_rate = down_sample_rate

        self.compare_bank = compare_bank.cuda() # meta-gaussians
        self.meta_gaussian_class_indices = meta_gaussian_class_indices
        self.meta_gaussian_indices = None

    def forward(self, predicted_features, box_obj=None, mask_out_padded=False):
        if mask_out_padded is True and box_obj is not None:
            try: 
                box_obj = bbt.from_numpy(box_obj.squeeze(0).numpy())
            except Exception:
                print("Illegal boundary box!")
                return None
            object_height, object_width = box_obj[0], box_obj[1]
            object_height = (
                object_height[0] // self.down_sample_rate,
                object_height[1] // self.down_sample_rate,
            )
            h0, h1 = object_height
            if h0 == h1:
                object_height = (h0, h1 + 1)
            
            object_width = (
                object_width[0] // self.down_sample_rate,
                object_width[1] // self.down_sample_rate,
            )
            w0, w1 = object_width
            if w0 == w1:
                object_width = (w0, w1 + 1)

            predicted_features = predicted_features[
                ...,
                object_height[0] : object_height[1],
                object_width[0] : object_width[1],
            ]
        # Feature matching based on max over all gaussians, of shape [num_gaussians, num_features]
        score_per_pixel = torch.matmul(self.compare_bank, predicted_features.reshape(predicted_features.shape[1], -1))
        score_per_pixel = score_per_pixel / 2 + 0.5

        scores_val, score_idx = torch.max(score_per_pixel, dim=0)
        self.meta_gaussian_indices = score_idx
        return scores_val # shape of (num_features 8000)
        

class FeatureClassifierCAVE(nn.Module):
    def __init__(self, num_classes):
        super(FeatureClassifierCAVE, self).__init__()
        self.num_classes = num_classes

    def forward(self, scores_val, meta_gaussian_indices, meta_gaussian_class_indices):
        # For loop 
        scores = []
        score_class_indices = torch.tensor([meta_gaussian_class_indices[idx] for idx in meta_gaussian_indices])
        for cls_idx in range(self.num_classes):
            score_to_keep = torch.where(score_class_indices == cls_idx)
            score = torch.sum(scores_val[score_to_keep]) / meta_gaussian_class_indices.shape[0]
            scores.append(score)
        scores = torch.stack(scores)
        return scores

class CAVENetE2E(nn.Module):
    def __init__(self, feature_extractor: NetE2E, meta_fbank, class_labels, down_sample_rate=1):
        super(CAVENetE2E, self).__init__()
        self.feature_extractor = feature_extractor.cuda() # Backbone model
        self.feature_matcher = FeatureMatchingCAVE(meta_fbank.features.cuda(), meta_fbank.class_indices, class_labels, down_sample_rate=down_sample_rate)
        self.classifier = FeatureClassifierCAVE(len(class_labels)).cuda()
        self.meta_gaussian_class_indices = meta_fbank.class_indices

    def forward(self, img, box_obj=None, mask_out_padded=False):
        # Extract features from backbone NetE2E
        predicted_features = self.feature_extractor.forward(img)
        # best feature-gaussian matching
        scores_val = self.feature_matcher(predicted_features, box_obj=box_obj, mask_out_padded=mask_out_padded)
        if scores_val is None:
            return None
        # Classify based on extracted features
        scores = self.classifier(scores_val, self.feature_matcher.meta_gaussian_indices, self.feature_matcher.meta_gaussian_class_indices)

        return scores # only add scores_val for grad_cam
  