"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from third_party.LRP_for_ResNet.src.lrp_filter import relevance_filter
from src.models.UpsamplingLayer import DoubleConv, Up
from src.models.KeypointRepresentationNet import Out
from src.models.NovumNetE2E import FeatureMatching, FeatureClassifier
import torch.nn.functional as F
import BboxTools as bbt


class RelevancePropagationBasicBlock(nn.Module):
    def __init__(
        self,
        layer: BasicBlock,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = [
            layer.conv1,
            layer.bn1,
            layer.relu,
            layer.conv2,
            layer.bn2,
        ]
        self.downsample = layer.downsample
        self.relu = layer.relu
        self.eps = eps
        self.top_k = top_k

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())

    def mainstream_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

        activations.pop()  # ignore output of this basic block
        activations = [a.data.requires_grad_(True) for a in activations]

        # NOW, IGNORES DOWN-SAMPLING & SKIP CONNECTION
        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        return r_in

    def shortcut_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return r
        a = a.data.requires_grad_(True)
        assert isinstance(self.downsample[0], nn.Conv2d)
        return RelevancePropagationConv2d(self.downsample[0], eps=self.eps, top_k=self.top_k)(a, r)

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        ratio = self._calc_mainstream_flow_ratio(a)
        assert r.shape == ratio.shape
        r_mainstream = ratio * r
        r_shortcut = (1 - ratio) * r
        r_mainstream = self.mainstream_backward(a, r_mainstream)
        r_shortcut = self.shortcut_backward(a, r_shortcut)
        return r_mainstream + r_shortcut


class RelevancePropagationBasicBlockSimple(RelevancePropagationBasicBlock):
    """ Relevance propagation for BasicBlock Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: BasicBlock, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationBasicBlockFlowsPureSkip(RelevancePropagationBasicBlock):
    """ Relevance propagation for BasicBlock Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: BasicBlock, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())


class RelevancePropagationBasicBlockSimpleFlowsPureSkip(RelevancePropagationBasicBlock):
    """ Relevance propagation for BasicBlock Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: BasicBlock, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.full_like(input_, 0.5)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationBottleneck(nn.Module):
    def __init__(
        self,
        layer: Bottleneck,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = [
            layer.conv1,
            layer.bn1,
            layer.relu,
            layer.conv2,
            layer.bn2,
            layer.relu,
            layer.conv3,
            layer.bn3,
        ]
        self.downsample = layer.downsample
        self.relu = layer.relu
        self.eps = eps
        self.top_k = top_k

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())

    def mainstream_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

        activations.pop()  # ignore output of this bottleneck block
        activations = [a.data.requires_grad_(True) for a in activations]

        # NOW, IGNORES DOWN-SAMPLING & SKIP CONNECTION
        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        return r_in

    def shortcut_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return r
        a = a.data.requires_grad_(True)
        assert isinstance(self.downsample[0], nn.Conv2d)
        return RelevancePropagationConv2d(self.downsample[0], eps=self.eps, top_k=self.top_k)(a, r)

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        ratio = self._calc_mainstream_flow_ratio(a)
        assert r.shape == ratio.shape
        r_mainstream = ratio * r
        r_shortcut = (1 - ratio) * r
        r_mainstream = self.mainstream_backward(a, r_mainstream)
        r_shortcut = self.shortcut_backward(a, r_shortcut)
        return r_mainstream + r_shortcut


class RelevancePropagationBottleneckSimple(RelevancePropagationBottleneck):
    """ Relevance propagation for Bottleneck Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: Bottleneck, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationBottleneckFlowsPureSkip(RelevancePropagationBottleneck):
    """ Relevance propagation for Bottleneck Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: Bottleneck, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())


class RelevancePropagationBottleneckSimpleFlowsPureSkip(RelevancePropagationBottleneck):
    """ Relevance propagation for Bottleneck Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: Bottleneck, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.full_like(input_, 0.5)
        return torch.full_like(self.downsample(input_), 0.5)


# NOVUM and CAVE specific layers
class RelevancePropagationDoubleConv(nn.Module):
    """
    Relevance propagation for DoubleConv Layer
    """
    def __init__(self, layer: DoubleConv, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layers = [ l for l in layer.doubleconv ]
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))
        activations.pop()
        activations = [a.data.requires_grad_(True) for a in activations]

        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        return r_in

    
class RelevancePropagationUp(nn.Module):
    """
    Relevance propagation for Up layer
    """
    def __init__(self, layer: Up, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layers = [layer.up, layer.doubleconv]
        # self.layers.extend([l for l in layer.doubleconv.doubleconv])
        self.eps = eps
        self.top_k = top_k

    def forward(self, a1: torch.Tensor, a2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            activations = [a1]
            for layer in self.layers:
                if isinstance(layer, nn.Upsample):
                    a1 = layer.forward(activations[-1])
                    diffY = torch.tensor([a2.size()[2] - a1.size()[2]])
                    diffX = torch.tensor([a2.size()[3] - a1.size()[3]])
                    a1 =  F.pad(a1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]) # L, R, T, B
                    a = torch.cat([a2, a1], dim=1)
                    activations.append(a)
                else:
                    activations.append(activations[-1])
        activations.pop()
        activations = [a.data.requires_grad_(True) for a in activations]

        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, DoubleConv):
                r_in = RelevancePropagationDoubleConv(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            elif isinstance(layer, nn.Upsample):
                r_shortcut, r_out = torch.split(r_out, [a2.size(1), a1.size(1)], dim=1)
                r_out = r_out[:, :, diffY // 2 : diffY // 2 + a1.size(2), diffX // 2 : diffX // 2 + a1.size(3)]
                r_in = RelevancePropagationUpsample(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            else:
                raise NotImplementedError
            r_out = r_in
        return r_in, r_shortcut


class RelevancePropagationOut(nn.Module):
    """
    Relevance propagation for Out layer
    """
    def __init__(self, layer: Out, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data

        return r

####################################################
#                       NOVUM                      #
####################################################
class RelevancePropagationFeatureMatching(nn.Module):
    """
    Relevance propagation for FeatureMatching layer
    """
    def __init__(self, layer: FeatureMatching, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor, box_obj=None, mask_out_padded=False) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k) # 8000
        
        self.layer.forward(a, box_obj=box_obj, mask_out_padded=mask_out_padded) # need to forward to obtain the corresponding gaussians

        if box_obj is not None and mask_out_padded:
            box_obj = bbt.from_numpy(box_obj.squeeze(0).numpy())
            object_height, object_width = box_obj[0], box_obj[1]
            object_height = (
                object_height[0] // self.layer.down_sample_rate,
                object_height[1] // self.layer.down_sample_rate,
            )
            object_width = (
                object_width[0] // self.layer.down_sample_rate,
                object_width[1] // self.layer.down_sample_rate,
            )
            f_i = a[..., object_height[0] : object_height[1], object_width[0] : object_width[1]]
            f_i = f_i.reshape(f_i.shape[1], -1).permute(1, 0)
        else:
            f_i = a.reshape(a.shape[1], -1).permute(1, 0)

        c_k = self.layer.compare_bank[self.layer.gaussian_indices] # gaussians corresponding to features
        s = f_i * c_k # alignment scores along channel dimension (8000, 128)

        # distribute relevance along channel dimension
        s = s / ( s.sum(dim=1, keepdim=True) + 1e-8)
        r = r.unsqueeze(1) * s

        if box_obj is not None and mask_out_padded:
            r_out = torch.zeros_like(a) # [1, 128, 80, 100]
            r_out[..., 
                  object_height[0] : object_height[1], 
                  object_width[0] : object_width[1]
                  ] = r.permute(1, 0).reshape(a.shape[0], 
                                              a.shape[1], 
                                              object_height[1] - object_height[0], 
                                              object_width[1] - object_width[0])
        else:
            r_out = r.permute(1, 0).reshape(a.shape)
        
        return r_out

class RelevancePropagationFeatureClassifier(nn.Module):
    """
    Relevance propagation for FeatureClassifier layer
    """
    def __init__(self, layer: FeatureClassifier, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, score_idx: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a, score_idx) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


####################################################
#                       CAVE                       #
####################################################
class RelevancePropagationFeatureMatchingCave(nn.Module):
    """
    Relevance propagation for FeatureMatching layer
    """
    def __init__(self, layer: FeatureMatching, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor, box_obj=None, mask_out_padded=False) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k) # 8000
        
        self.layer.forward(a, box_obj=box_obj, mask_out_padded=mask_out_padded) # need to forward to obtain the corresponding gaussians

        if box_obj is not None and mask_out_padded:
            box_obj = bbt.from_numpy(box_obj.squeeze(0).numpy())
            object_height, object_width = box_obj[0], box_obj[1]
            object_height = (
                object_height[0] // self.layer.down_sample_rate,
                object_height[1] // self.layer.down_sample_rate,
            )
            object_width = (
                object_width[0] // self.layer.down_sample_rate,
                object_width[1] // self.layer.down_sample_rate,
            )
            f_i = a[..., object_height[0] : object_height[1], object_width[0] : object_width[1]]
            f_i = f_i.reshape(f_i.shape[1], -1).permute(1, 0)
        else:
            f_i = a.reshape(a.shape[1], -1).permute(1, 0)

        c_k = self.layer.compare_bank[self.layer.meta_gaussian_indices] # gaussians corresponding to features
        s = f_i * c_k # alignment scores along channel dimension (num_features, 128)

        # distribute relevance along channel dimension
        s = s / ( s.sum(dim=1, keepdim=True) + 1e-8)
        r = r.unsqueeze(1) * s
        if box_obj is not None and mask_out_padded:
            r_out = torch.zeros_like(a) # [1, 128, 80, 100]
            r_out[..., 
                  object_height[0] : object_height[1], 
                  object_width[0] : object_width[1]
                  ] = r.permute(1, 0).reshape(a.shape[0], 
                                              a.shape[1], 
                                              object_height[1] - object_height[0], 
                                              object_width[1] - object_width[0])
        else:
            r_out = r.permute(1, 0).reshape(a.shape)
        print("r_out:", r_out.shape)
        return r_out

class RelevancePropagationFeatureClassifierCave(nn.Module):
    """
    Relevance propagation for FeatureClassifier layer of Cave
    """
    def __init__(self, layer: FeatureClassifier, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, meta_gaussian_indices: torch.Tensor, meta_gaussian_class_indices: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a, meta_gaussian_indices, meta_gaussian_class_indices) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

class RelevancePropagationUpsample(nn.Module):
    def __init__(self, layer: torch.nn.Upsample, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r

    
class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.AdaptiveAvgPool2d,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05, top_k: float = 0.0
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        #TODO: double check if the relevance is conserved
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.MaxPool2d,
        mode: str = "avg",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        # print(f"maxpool2d {r.min()}, {r.max()}")
        return r


class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Conv2d,
        mode: str = "z_plus",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        # print(f"before norm: {r.sum()}")
        # r = (r - r.min()) / (r.max() - r.min())
        # print(f"after norm: {r.sum()}\n")
        if r.shape != a.shape:
            raise RuntimeError("r.shape != a.shape")
        return r


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Linear,
        mode: str = "z_plus",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        # print(f"Linear {r.min()}, {r.max()}")
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationBatchNorm2d(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.BatchNorm2d, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer: nn.Module, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r
