import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .UpsamplingLayer import DoubleConv
from .UpsamplingLayer import Up

net_stride = {
    "resnetext": 8,
}
net_out_dimension = {
    "resnetext": 256,
}

class ResNetExt(nn.Module):
    def __init__(self, pretrained, nb_classes=12, pose_output_size=126, dropout_p=None):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential()
        self.extractor.add_module("0", net.conv1)
        self.extractor.add_module("1", net.bn1)
        self.extractor.add_module("2", net.relu)
        self.extractor.add_module("3", net.maxpool)
        self.extractor.add_module("4", net.layer1)
        self.extractor.add_module("5", net.layer2)
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(2048, 1024, 512)
        self.upsample2 = Up(1024, 512, 256)
        self.with_dropout = dropout_p is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x1 = self.extractor(x)
        x2 = self.extractor1(x1)
        x3 = self.extractor2(x2)
        features = self.upsample2(self.upsample1(self.upsample0(x3), x2), x1)
        if self.with_dropout:
            features = self.dropout(features)
        return features

    def freeze_features_extractor(self):
        for param in self.extractor.parameters():
            param.requires_grad = False
        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def unfreeze_features_extractor(self):
        for param in self.extractor.parameters():
            param.requires_grad = True
        for param in self.extractor1.parameters():
            param.requires_grad = True
        for param in self.extractor2.parameters():
            param.requires_grad = True


def resnetext(pretrain):
    net = ResNetExt(pretrained=pretrain)
    return net


def keypoints_to_pixel_index(keypoints, downsample_rate, original_img_size=(480, 640)):
    line_size = original_img_size[1] // downsample_rate
    return (
        keypoints[:, :, 0] // downsample_rate * line_size
        + keypoints[:, :, 1] // downsample_rate
    )


def get_noise_pixel_index(keypoints, max_size, n_samples, obj_mask=None):
    n = keypoints.shape[0]
    # remove the point in keypoints by set probability to 0 otherwise 1 -> mask [n, size] with 0 or 1
    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    # if keypoints is negative, set it to 0
    keypoints = torch.clamp(keypoints, min=0)
    mask = mask.scatter(1, keypoints.type(torch.long), 0.0)
    if obj_mask is not None:
        mask *= obj_mask
    # generate the sample by the probabilities
    return torch.multinomial(mask, n_samples)

class GlobalLocalConverter(nn.Module):
    def __init__(self, local_size):
        super().__init__()
        self.local_size = local_size
        self.padding = sum(([t - 1 - t // 2, t // 2] for t in local_size[::-1]), [])

    def forward(self, X):
        n, c, h, w = X.shape  # torch.Size([1, 2048, 8, 8])

        # N, C, H, W -> N, C, H + local_size0 - 1, W + local_size1 - 1
        X = F.pad(X, self.padding)

        # N, C, H + local_size0 - 1, W + local_size1 - 1 -> N, C * local_size0 * local_size1, H * W
        X = F.unfold(X, kernel_size=self.local_size)

        # N, C * local_size0 * local_size1, H * W -> N, C, local_size0, local_size1, H * W
        # X = X.view(n, c, *self.local_size, -1)

        # X:  N, C * local_size0 * local_size1, H * W
        return X

def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b * e * f
    return out


class NetE2E(nn.Module):
    def __init__(
        self,
        pretrain,
        net_type,
        local_size,
        output_dimension,
        n_noise_points=0,
        noise_on_mask=True,
    ):
        # output_dimension = 128
        super().__init__()
        if net_type == "resnetext":
            self.net = resnetext(pretrain)
        else:
            raise ValueError("Unknown net type")

        self.size_number = local_size[0] * local_size[1]
        self.output_dimension = output_dimension
        self.net_type = net_type
        self.net_stride = net_stride[net_type]
        self.converter = GlobalLocalConverter(local_size)
        self.noise_on_mask = noise_on_mask

        self.out_layer = nn.Linear(
            net_out_dimension[net_type] * self.size_number,
            self.output_dimension,
        )

        self.n_noise_points = n_noise_points

    # forward
    def forward_test(self, X):
        X = self.net.forward(X)

        if self.output_dimension == -1:
            return F.normalize(X, p=2, dim=1)
        if self.size_number == 1:
            X = torch.nn.functional.conv2d(
                X,
                self.out_layer.weight.unsqueeze(2).unsqueeze(3),
            )
        elif self.size_number > 1:
            X = torch.nn.functional.conv2d(
                X,
                self.out_layer.weight.view(
                    self.output_dimension,
                    net_out_dimension[self.net_type],
                    self.size_number,
                )
                .permute(2, 0, 1)
                .reshape(
                    self.size_number * self.output_dimension,
                    net_out_dimension[self.net_type],
                )
                .unsqueeze(2)
                .unsqueeze(3),
            )
        # n, c, w, h
        # 1, 128, (w_original - 1) // 32 + 1, (h_original - 1) // 32 + 1
        return F.normalize(X, p=2, dim=1)

    def forward(self, X, keypoint_positions, obj_mask=None, return_map=False):
        with open(os.path.join("/BS/bcos-craft/work/NOVUM/logs/train/ImageNet3D/ellipsoid", f"log_forward.txt"), "a") as f:
            f.write(
                f"[GPU {torch.cuda.current_device()}] Local batch size: {X.shape[0]}\n"
            )
        # X=torch.ones(1, 3, 224, 300), kps = torch.tensor([[(36, 40), (90, 80)]])
        # n images, k keypoints and 2 states.
        # Keypoint input -> n * k * 2 (k keypoints for n images) (must be position on original image)
        n = X.shape[0]  # n = 1
        img_shape = X.shape[2::]

        # downsample_rate = 32
        m = self.net.forward(X)

        # N, C * local_size0 * local_size1, H * W
        X = self.converter(m)

        keypoint_idx = keypoints_to_pixel_index(
            keypoints=keypoint_positions,
            downsample_rate=self.net_stride,
            original_img_size=img_shape,
        ).type(torch.long)

        if self.n_noise_points == 0:
            keypoint_all = keypoint_idx
        else:
            if obj_mask is not None:
                obj_mask = F.max_pool2d(
                    obj_mask.unsqueeze(dim=1),
                    kernel_size=self.net_stride,
                    stride=self.net_stride,
                    padding=(self.net_stride - 1) // 2,
                )
                obj_mask = obj_mask.view(obj_mask.shape[0], -1)
                assert obj_mask.shape[1] == X.shape[2], (
                    "mask_: " + str(obj_mask.shape) + " fearture_: " + str(X.shape)
                )
            if self.noise_on_mask:
                keypoint_noise = get_noise_pixel_index(
                    keypoint_idx,
                    max_size=X.shape[2],
                    n_samples=self.n_noise_points,
                    obj_mask=obj_mask,
                )
            else:
                keypoint_noise = get_noise_pixel_index(
                    keypoint_idx,
                    max_size=X.shape[2],
                    n_samples=self.n_noise_points,
                    obj_mask=None,
                )
            keypoint_all = torch.cat((keypoint_idx, keypoint_noise), dim=1)

        # N, C * local_size0 * local_size1, H * W -> N, H * W, C * local_size0 * local_size1
        X = torch.transpose(X, 1, 2)

        # check that keypoints_all are in the range of X.shape
        keypoint_all[keypoint_all >= X.shape[1]] = 0
        keypoint_all[keypoint_all < 0] = 0

        # N, H * W, C * local_size0 * local_size1 -> N, keypoint_all, C * local_size0 * local_size1
        X = batched_index_select(X, dim=1, inds=keypoint_all)

        # L2norm, fc layer, -> dim along d
        if self.out_layer is None:
            X = F.normalize(X, p=2, dim=2)
            X = X.view(n, -1, net_out_dimension[self.net_type])
        else:
            X = F.normalize(self.out_layer(X), p=2, dim=2)
            X = X.view(n, -1, self.out_layer.weight.shape[0])

        # n * k * output_dimension
        if return_map:
            return X, F.normalize(
                torch.nn.functional.conv2d(
                    m,
                    self.out_layer.weight.unsqueeze(2).unsqueeze(3),
                ),
                p=2,
                dim=1,
            )
        return X

    def cuda(self, device=None):
        self.net.cuda(device=device)
        self.converter.cuda(device=device)
        self.out_layer.cuda(device=device)
        return self
