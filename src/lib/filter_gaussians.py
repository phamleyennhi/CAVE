import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

import torch
from src.lib.config import load_config, parse_args
from src.lib.get_n_list import compute_max_n


def show_grid(save_path:str, cls_indices: list, cls_labels: list):
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(os.path.join(save_path, f"{cls_indices[i]}_{cls_labels[i]}.png"))
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "overall.png"))

def plot_visibility_distribution(result_path: str, save_path: str, cls_idx: int, cls_label: str):
    iskpvisible_path = os.path.join(result_path, f"{cls_idx}_{cls_label}_kpvisible.pt")
    all_iskpvisible = torch.load(iskpvisible_path)
    
    df = pd.DataFrame(all_iskpvisible["count"].numpy(), columns=["Visibility Count"])
    max_count = all_iskpvisible["max_count"]

    plt.figure(figsize=(15, 6))
    plot = sns.displot(data=df, x="Visibility Count")
    plot.set_axis_labels("Visibility Count", "No.Gaussians")
    plot.fig.suptitle(f"C{cls_idx}.{cls_label} of {max_count} train samples\n")
    plt.savefig(os.path.join(save_path, f"{cls_idx}_{cls_label}.png"))

def plot_gaussian_feature_scores(scores: torch.Tensor, filter_gaussians: torch.Tensor, cls_idx: int, cls_label: int, num_gaussians: int, save_path:str, bw_adjust: float = 1.0, vis_threshold: float = 0.1):
    num_gaussians = scores.shape[0]
    num_out_gaussians = num_gaussians - torch.sum(filter_gaussians)
    plt.figure(figsize=(10, 6))

    for i in range(num_gaussians):
        if filter_gaussians[i] == 1:
            sns.kdeplot(scores[i].numpy(), alpha=0.1, clip=(-1, 1), bw_adjust=bw_adjust)
    
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title(f"Overlayed Distributions of Cosine Similarities across {int(num_gaussians - num_out_gaussians)}/{num_gaussians} Gaussian for class {cls_idx} - {cls_label}\n Visibility Threshold: {vis_threshold}")

    plt.savefig(os.path.join(save_path, f"{cls_idx}_{cls_label}.png"))


def filter_gaussians_by_visibility(result_path: str, cls_idx: int, cls_label: str, vis_threshold: float = 0.1):
        iskpvisible_path = os.path.join(result_path, f"{cls_idx}_{cls_label}_kpvisible.pt")
        all_iskpvisible = torch.load(iskpvisible_path)

        count = all_iskpvisible["count"]
        max_count = all_iskpvisible["max_count"]
        print("min_count:", torch.min(count), "max_count:", torch.max(count))
        print("No. training samples:", max_count)
        filter_gaussians = torch.where(count > int(max_count * vis_threshold), 1, 0)
        print("CLASS", cls_idx, cls_label, "remove:", filter_gaussians.shape[0] - torch.sum(filter_gaussians), "keep:", torch.sum(filter_gaussians))
        return filter_gaussians
        

def filter_gaussians_by_score(result_path: str, cls_idx: int, cls_label: str, score_threshold=0.9, top_k=100):
    # load scores and indices
    score_path = os.path.join(result_path, f"{cls_idx}_{cls_label}_scores.pt")
    indices_path = os.path.join(result_path, f"{cls_idx}_{cls_label}_indices.pt")
    all_scores_val = torch.load(score_path)
    all_scores_idx = torch.load(indices_path)

    # compute top scores per Gaussian
    topk_gaussian_scores, topk_gaussian_indices = torch.topk(all_scores_val, dim=1, k=top_k)

    # filter out unqualified Gaussians C_k
    count_C_k = 0
    for C_k, top_scores in enumerate(tqdm(topk_gaussian_scores)):
        if top_scores[-1] < score_threshold:
            count_C_k += 1
    print(f"Class {self.cls_idx} - {self.cls_label}, topK {top_k}, threshold {score_threshold}:")
    print(f"Total number of disqualified gaussians C_k:", count_C_k)
    print(f"Total number of qualified gaussians C_k:", self.n_list_set[self.cls_idx] - count_C_k)
    print()

    return 0

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args, load_default_config=False, log_info=False)

    max_n, n_list_set = compute_max_n(config)
    result_path = os.path.join(config.zoom_out.paths.root, config.zoom_out.paths.match_feature_scores)
    kpvis_threshold = config.zoom_out.kpvis_threshold


    # choose cuboid class
    class_labels = config.dataset.classes

    # set up paths
    vis_path = os.path.join(config.zoom_out.paths.root, config.zoom_out.paths.kpvisible_distribution)
    # plot overview 
    cls_indices = list(range(len(class_labels)))


    cuboid_cls_idx = 11
    cuboid_cls_label = class_labels[cuboid_cls_idx]

    score_distribution_path = os.path.join(config.zoom_out.paths.root, config.zoom_out.paths.score_distribution, f"cuboid_{cuboid_cls_idx}_{cuboid_cls_label}", str(kpvis_threshold))
    os.makedirs(score_distribution_path, exist_ok=True)

    for cls_idx, cls_label in enumerate(class_labels):
        scores = torch.load(os.path.join(result_path, f"cuboid_{cuboid_cls_idx}_{cuboid_cls_label}", f"{cls_idx}_{cls_label}_scores.pt"))
        if cls_idx == cuboid_cls_idx:
            filter_gaussians = filter_gaussians_by_visibility(result_path, cls_idx, cls_label, vis_threshold=0.1)

        else:
            kpvis_threshold = 0.0
            filter_gaussians = torch.ones(n_list_set[cuboid_cls_idx])

        # plot_visibility_distribution(result_path, os.path.join(config.zoom_out.paths.root, config.zoom_out.paths.kpvisible_distribution), cls_idx, cls_label)
        # plot score distribution based on kpvis threshold
        plot_gaussian_feature_scores(scores, filter_gaussians, cls_idx, cls_label, n_list_set[cuboid_cls_idx], score_distribution_path, vis_threshold=kpvis_threshold)
    
    show_grid(score_distribution_path, cls_indices, class_labels)
        