import os
import torch
import numpy as np

class MetaFeatureBank():
    #TODO: include background features
    def __init__(self, cls_labels, fbank, merging_labels_path, aggregate_by="mean", num_clusters=20):

        self.cls_labels = cls_labels
        self.nb_classes = len(self.cls_labels)
        self.fbank = fbank # all gaussian features
        self.merging_labels_path = merging_labels_path
        self.aggregate_by = aggregate_by
        self.num_clusters = num_clusters

        self.memory = []
        self.memory_gaussian_indices = []
        self.memory_class_indices = []
    
    @property
    def features(self):
        return self.memory
    
    @property
    def class_indices(self):
        return self.memory_class_indices

    @property
    def gaussian_indices(self):
        return self.memory_gaussian_indices
    
    def compute_meta_gaussians(self, n_list_set, seed=""):
        if self.aggregate_by == "mean" or self.aggregate_by == "sum":
            for cls_idx, cls_label in enumerate(self.cls_labels):
                # class gaussians
                gaussian_features = self.fbank.features_of(cls_idx, n_list_set)
                cls_gaussian_indices = []
                # assign cluster
                cluster_labels = torch.load(os.path.join(self.merging_labels_path, f"{cls_idx}_{cls_label}", f"labels{seed}.pt"))
                cluster_labels = torch.tensor(cluster_labels)
                unique_labels = torch.unique(cluster_labels)
                # compute cluster
                for label in unique_labels:
                    if label == -1:
                        continue
                    
                    # corresponding gaussian indices for the cluster
                    label_indices = torch.where(cluster_labels == label)[0]
                    if label_indices.shape[0] > 0:
                        cls_gaussian_indices.append(label_indices)

                    cluster_features = gaussian_features[label_indices]
                    if self.aggregate_by == "mean":
                        representative_feature = cluster_features.mean(dim=0)
                    elif self.aggregate_by == "sum":
                        representative_feature = cluster_features.sum(dim=0)
                    else:
                        raise NotImplementedError(f"Not implemented for aggregate_by {self.aggregate_by}")
                    self.memory.append(representative_feature)
                # self.memory_class_indices.extend((len(unique_labels)-1) * [cls_idx])
                self.memory_class_indices.extend(self.num_clusters * [cls_idx])
                self.memory_gaussian_indices.append(cls_gaussian_indices)
            
        elif self.aggregate_by is None:
            for cls_idx, cls_label in enumerate(self.cls_labels):
                concepts = torch.load(os.path.join(self.merging_labels_path, f"{cls_idx}_{cls_label}", f"concepts{seed}.pt"))
                self.memory.extend(torch.tensor(concepts))
                self.memory_class_indices.extend(len(concepts) * [cls_idx])
        else:
            raise NotImplementedError(f"Not implemented for {self.aggregate_by}")
        
        self.memory = torch.stack(self.memory)
        self.memory_class_indices = torch.tensor(self.memory_class_indices)
            