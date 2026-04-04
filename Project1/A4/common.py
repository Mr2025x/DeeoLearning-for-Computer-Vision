"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        for level_name, feature_shape in dummy_out_shapes:
            in_channels = feature_shape[1]
            # 这里统一命名为 c3_lateral, c3_output 等
            self.fpn_params[f"{level_name}_lateral"] = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

            self.fpn_params[f"{level_name}_output"] = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################
        m5 = self.fpn_params["c5_lateral"](backbone_feats["c5"])
        m4 = self.fpn_params["c4_lateral"](backbone_feats["c4"])
        m3 = self.fpn_params["c3_lateral"](backbone_feats["c3"])

        # 2. 自顶向下融合
        m4 = m4 + F.interpolate(m5, scale_factor=2, mode="nearest")
        m3 = m3 + F.interpolate(m4, scale_factor=2, mode="nearest")

        # 3. 输出平滑：严格使用上面定义的 c5_output 等名字
        fpn_feats["p5"] = self.fpn_params["c5_output"](m5)
        fpn_feats["p4"] = self.fpn_params["c4_output"](m4)
        fpn_feats["p3"] = self.fpn_params["c3_output"](m3)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        # 1. 解析当前特征图的高度 H 和宽度 W
        # feat_shape 的格式是 (B, C, H, W)
        H, W = feat_shape[2], feat_shape[3]

        # 2. 生成特征图上的 x 和 y 坐标轴序列
        # torch.arange 生成 [0, 1, 2, ..., W-1] 和 [0, 1, 2, ..., H-1]
        # 按照公式：加上 0.5 并乘以 stride
        shifts_x = (torch.arange(W, dtype=dtype, device=device) + 0.5) * level_stride
        shifts_y = (torch.arange(H, dtype=dtype, device=device) + 0.5) * level_stride

        # 3. 使用 meshgrid 生成二维网格坐标
        # indexing="ij" 确保输出的 y 和 x 网格形状均为 (H, W)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # 4. 将网格展平，并将 x 和 y 组合在一起
        # flatten() 会把 (H, W) 变成一维的 (H * W)
        # torch.stack 沿着最后的维度把 x 和 y 拼起来，变成形状为 (H * W, 2) 的张量
        # 注意顺序：通常目标检测中坐标习惯用 (x, y) 格式，而不是 (y, x)
        shifts = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=-1)

        # 存入字典
        location_coords[level_name] = shifts
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    # 1. 提取所有框的坐标 (左上角 x1, y1; 右下角 x2, y2)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 2. 计算每一个框的面积
    areas = (x2 - x1) * (y2 - y1)

    # 3. 按照得分从高到低对索引进行排序
    # scores.sort() 返回两个值：排序后的结果，以及对应的原始索引 (order)
    _, order = scores.sort(0, descending=True)

    keep_boxes = [] # 用于存放最终保留下来的框的索引

    # 只要 order 里面还有框，就一直循环
    while order.numel() > 0:
        # 如果只剩最后一个框了，直接保留并结束循环
        if order.numel() == 1:
            i = order[0].item()
            keep_boxes.append(i)
            break

        # 4. 每次循环挑出最高分的框索引 i，加入保留名单
        i = order[0].item()
        keep_boxes.append(i)

        # 5. 计算当前最高分框 (i) 与剩余所有框 (order[1:]) 的交集坐标
        # 使用 torch.max 和 torch.min 进行逐元素比较 (向量化计算)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        # 6. 计算交集的宽 (w) 和 高 (h)
        # 用 clamp(..., min=0) 保证如果两个框不相交，宽和高就是 0，而不是负数
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        # 7. 计算 IoU
        # 交集面积 / (框 i 的面积 + 剩余框的面积 - 交集面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 8. 过滤掉 IoU > iou_threshold 的框
        # 找出那些 IoU <= 阈值的框的索引 (也就是可以幸存下来的框)
        inds = torch.where(iou <= iou_threshold)[0]

        # 9. 更新 order，只保留幸存下来的框，进入下一轮循环
        # 注意：因为我们刚才计算 IoU 时没有包含 i 本身，用的是 order[1:]
        # 所以这里的 inds 对应的其实是 order[1:] 里的位置，我们需要加 1 来对齐原始的 order
        order = order[inds + 1]

    # 将 Python list 转换为要求返回的 PyTorch Tensor
    keep = torch.tensor(keep_boxes, dtype=torch.long, device=boxes.device)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
