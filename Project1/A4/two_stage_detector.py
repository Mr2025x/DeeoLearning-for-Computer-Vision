import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from a4_helper import *
from common import class_spec_nms, get_fpn_location_coords, nms
from torch import nn
from torch.nn import functional as F

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")

class RPNPredictionNetwork(nn.Module):
    """
    RPN prediction network that accepts FPN feature maps from different levels
    and makes two predictions for every anchor: objectness and box deltas.

    Faster R-CNN typically uses (p2, p3, p4, p5) feature maps. We will exclude
    p2 for have a small enough model for Colab.

    Conceptually this module is quite similar to `FCOSPredictionNetwork`.
    """

    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        """
        Args:
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
            num_anchors: Number of anchor boxes assumed per location (say, `A`).
                Faster R-CNN without an FPN uses `A = 9`, anchors with three
                different sizes and aspect ratios. With FPN, it is more common
                to have a fixed size dependent on the stride of FPN level, hence
                `A = 3` is default - with three aspect ratios.
        """
        super().__init__()

        self.num_anchors = num_anchors
        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. RPN shares this stem for objectness and box
        # regression (unlike FCOS, that uses separate stems).
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        stem_rpn = []
        # Replace "pass" statement with your code
        stem_rpn = []
        prev_channels = in_channels
        
        for curr_channels in stem_channels:
            conv = nn.Conv2d(prev_channels, curr_channels, kernel_size=3, stride=1, padding=1)
            # 权重和偏置初始化
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.0)
            
            stem_rpn.append(conv)
            stem_rpn.append(nn.ReLU())
            prev_channels = curr_channels

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_rpn = nn.Sequential(*stem_rpn)
        ######################################################################
        # TODO: Create TWO 1x1 conv layers for individually to predict
        # objectness and box deltas for every anchor, at every location.
        #
        # Objectness is obtained by applying sigmoid to its logits. However,
        # DO NOT initialize a sigmoid module here. PyTorch loss functions have
        # numerically stable implementations with logits.
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_obj = None  # Objectness conv
        self.pred_box = None  # Box regression conv

        # Replace "pass" statement with your code
        # Objectness: 每个位置输出 A 个分数
        self.pred_obj = nn.Conv2d(stem_channels[-1], self.num_anchors, kernel_size=1)
        
        # Box Regression: 每个位置输出 A 个 Anchor 的 4 个偏移量，共 4*A 个值
        self.pred_box = nn.Conv2d(stem_channels[-1], self.num_anchors * 4, kernel_size=1)
        
        # 别忘了对这两个 1x1 卷积也做相同的初始化
        nn.init.normal_(self.pred_obj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pred_obj.bias, 0.0)
        nn.init.normal_(self.pred_box.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pred_box.bias, 0.0)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict desired quantities for every anchor
        at every location. Format the output tensors such that feature height,
        width, and number of anchors are collapsed into a single dimension (see
        description below in "Returns" section) this is convenient for computing
        loss and perforning inference.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}.
                Each tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Objectness logits:     `(batch_size, H * W * num_anchors)`
            2. Box regression deltas: `(batch_size, H * W * num_anchors, 4)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. DO NOT apply sigmoid to objectness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        object_logits = {}
        boxreg_deltas = {}

        # Replace "pass" statement with your code
        for level_name, feature in feats_per_fpn_level.items():
            B, _, H, W = feature.shape
            
            # 1. 穿过共享的 Stem 网络
            stem_features = self.stem_rpn(feature)
            
            # 2. 获取初始预测 (形状为 [B, Channels, H, W])
            obj_logits_raw = self.pred_obj(stem_features)  # 形状: [B, A, H, W]
            box_deltas_raw = self.pred_box(stem_features)  # 形状: [B, A*4, H, W]
            
            # 3. 形状重组 (极其关键)
            # Objectness: [B, A, H, W] -> [B, H, W, A] -> [B, H*W*A]
            obj_logits_reshaped = obj_logits_raw.permute(0, 2, 3, 1).reshape(B, -1)
            
            # Box deltas: [B, A*4, H, W] -> [B, H, W, A*4] -> [B, H*W*A, 4]
            # 这里最后多出一个 4 的维度，代表每个 anchor 的 dx, dy, dw, dh
            box_deltas_reshaped = box_deltas_raw.permute(0, 2, 3, 1).reshape(B, -1, 4)
            
            # 存入字典
            object_logits[level_name] = obj_logits_reshaped
            boxreg_deltas[level_name] = box_deltas_reshaped
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [object_logits, boxreg_deltas]


@torch.no_grad()
def generate_fpn_anchors(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    stride_scale: int,
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
):
    """
    Generate multiple anchor boxes at every location of FPN level. Anchor boxes
    should be in XYXY format and they should be centered at the given locations.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H, W is the size of FPN feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        stride_scale: Size of square anchor at every FPN levels will be
            `(this value) * (FPN level stride)`. Default is 4, which will make
            anchor boxes of size (32x32), (64x64), (128x128) for FPN levels
            p3, p4, and p5 respectively.
        aspect_ratios: Anchor aspect ratios to consider at every location. We
            consider anchor area to be `(stride_scale * FPN level stride) ** 2`
            and set new width and height of anchors at every location:
                new_width = sqrt(area / aspect ratio)
                new_height = area / new_width

    Returns:
        TensorDict
            Dictionary with same keys as `locations_per_fpn_level` and values as
            tensors of shape `(HWA, 4)` giving anchors for all locations
            per FPN level, each location having `A = len(aspect_ratios)` anchors.
            All anchors are in XYXY format and their centers align with locations.
    """

    # Set these to `(N, A, 4)` Tensors giving anchor boxes in XYXY format.
    anchors_per_fpn_level = {
        level_name: None for level_name, _ in locations_per_fpn_level.items()
    }

    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        # List of `A = len(aspect_ratios)` anchor boxes.
        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            ##################################################################
            # TODO: Implement logic for anchor boxes below. Write vectorized
            # implementation to generate anchors for a single aspect ratio.
            # Fill `anchor_boxes` list above.
            #
            # Calculate resulting width and height of the anchor box as per
            # `stride_scale` and `aspect_ratios` definitions. Then shift the
            # locations to get top-left and bottom-right co-ordinates.
            ##################################################################
            # Replace "pass" statement with your code
            x_c = locations[:, 0]
            y_c = locations[:, 1]
            
            # 2. 计算当前 FPN 层 Anchor 的基础面积
            area = (stride_scale * level_stride) ** 2
            
            # 3. 根据面积和当前的长宽比计算真实的宽和高
            import math
            new_width = math.sqrt(area / aspect_ratio)
            new_height = area / new_width
            
            # 4. 根据中心点和宽高，计算左上角 (x1, y1) 和 右下角 (x2, y2)
            # 注意：这里的 x_c, y_c 是长度为 N 的一维张量，标量 new_width/2 会自动广播
            x1 = x_c - new_width / 2.0
            y1 = y_c - new_height / 2.0
            x2 = x_c + new_width / 2.0
            y2 = y_c + new_height / 2.0
            
            # 5. 将这四个坐标沿着列方向(dim=1)拼接起来
            # 每一行代表一个 anchor 的 [x1, y1, x2, y2]
            # 最终 box_coords shape: (N, 4)
            box_coords = torch.stack([x1, y1, x2, y2], dim=1)
            
            # 6. 加入列表
            anchor_boxes.append(box_coords)
            
            ##################################################################
            #                           END OF YOUR CODE                     #
            ##################################################################

        # shape: (A, H * W, 4)
        anchor_boxes = torch.stack(anchor_boxes)
        # Bring `H * W` first and collapse those dimensions.
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes

    return anchors_per_fpn_level


@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection-over-union (IoU) between pairs of box tensors. Input
    box tensors must in XYXY format.

    Args:
        boxes1: Tensor of shape `(M, 4)` giving a set of box co-ordinates.
        boxes2: Tensor of shape `(N, 4)` giving another set of box co-ordinates.

    Returns:
        torch.Tensor
            Tensor of shape (M, N) with `iou[i, j]` giving IoU between i-th box
            in `boxes1` and j-th box in `boxes2`.
    """

    ##########################################################################
    # TODO: Implement the IoU function here.                                 #
    ##########################################################################
    # Replace "pass" statement with your code
    # 1. 计算两组框各自的面积
    # boxes shape: (M, 4) -> area shape: (M,)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    # boxes shape: (N, 4) -> area shape: (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 2. 利用广播机制计算交集框的左上角 (lt) 和右下角 (rb)
    # boxes1[:, None, :2] shape: (M, 1, 2)
    # boxes2[None, :, :2] shape: (1, N, 2)
    # torch.max broadcast 后的 lt shape: (M, N, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # 交集的 x1, y1
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # 交集的 x2, y2

    # 3. 计算交集的宽和高，如果框不相交，用 clamp 保证宽高等于 0
    # wh shape: (M, N, 2)
    wh = (rb - lt).clamp(min=0) 

    # 4. 计算交集面积
    # inter shape: (M, N)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # 5. 计算并集面积 (利用容斥原理和广播机制)
    # area1[:, None] shape (M, 1) + area2[None, :] shape (1, N) 
    # broadcast 后的 union shape: (M, N)
    union = area1[:, None] + area2[None, :] - inter

    # 6. 计算 IoU，并防止除以 0 的极小可能 (加上 1e-6 甚至可以省略，因为物理框通常面积大于0)
    iou = inter / union
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return iou


@torch.no_grad()
def rcnn_match_anchors_to_gt(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, float],
) -> TensorDict:
    """
    Match anchor boxes (or RPN proposals) with a set of GT boxes. Anchors having
    high IoU with any GT box are assigned "foreground" and matched with that box
    or vice-versa.

    NOTE: This function is NOT BATCHED. Call separately for GT boxes per image.

    Args:
        anchor_boxes: Anchor boxes (or RPN proposals). Dictionary of three keys
            a combined tensor of some shape `(N, 4)` where `N` are total anchors
            from all FPN levels, or a set of RPN proposals.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.
        iou_thresholds: Tuple of (low, high) IoU thresholds, both in [0, 1]
            giving thresholds to assign foreground/background anchors.
    """

    # Filter empty GT boxes:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]

    # If no GT boxes are available, match all anchors to background and return.
    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)

    # Match matrix => pairwise IoU of anchors (rows) and GT boxes (columns).
    # STUDENTS: This matching depends on your IoU implementation.
    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])

    # Find matched ground-truth instance per anchor:
    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]

    # Set boxes with low IoU threshold to background (-1).
    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1

    # Set remaining boxes to neutral (-1e8).
    neutral_idxs = (match_quality > iou_thresholds[0]) & (
        match_quality < iou_thresholds[1]
    )
    matched_gt_boxes[neutral_idxs, :] = -1e8
    return matched_gt_boxes


def rcnn_get_deltas_from_anchors(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Get box regression deltas that transform `anchors` to `gt_boxes`. These
    deltas will become GT targets for box regression. Unlike FCOS, the deltas
    are in `(dx, dy, dw, dh)` format that represent offsets to anchor centers
    and scaling factors for anchor size. Box regression is only supervised by
    foreground anchors. If GT boxes are "background/neutral", then deltas
    must be `(-1e8, -1e8, -1e8, -1e8)` (just some LARGE negative number).

    Follow Slide 68:
        https://web.eecs.umich.edu/~justincj/slides/eecs498/WI2022/598_WI2022_lecture13.pdf

    Args:
        anchors: Tensor of shape `(N, 4)` giving anchors boxes in XYXY format.
        gt_boxes: Tensor of shape `(N, 4)` giving matching GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving anchor deltas.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas.                               #
    # Remember to set the deltas of "background/neutral" GT boxes to -1e8    #
    ##########################################################################
    deltas = None
    # Replace "pass" statement with your code
    # 1. 识别出哪些是无效的 (background/neutral) 框
    # 在上一步中，无效框的坐标被填成了负数，所以只要 x1 < 0 就是无效框
    invalid_mask = gt_boxes[:, 0] < 0

    # 2. 将 Anchor 从 XYXY 转换为 CXCYWH
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0] + w_a / 2.0
    y_a = anchors[:, 1] + h_a / 2.0

    # 3. 将 GT Boxes 从 XYXY 转换为 CXCYWH
    # 注意：为了防止无效框产生负数的宽高导致后续 torch.log() 报错，
    # 我们用 .clamp(min=1.0) 强行让它的宽高等于 1 (反正是无效框，最后会被覆盖)
    w_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
    h_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
    x_gt = gt_boxes[:, 0] + w_gt / 2.0
    y_gt = gt_boxes[:, 1] + h_gt / 2.0

    # 4. 按照公式计算 Deltas
    dx = (x_gt - x_a) / w_a
    dy = (y_gt - y_a) / h_a
    dw = torch.log(w_gt / w_a)
    dh = torch.log(h_gt / h_a)

    # 5. 组合成 (N, 4) 的 Tensor
    deltas = torch.stack([dx, dy, dw, dh], dim=1)

    # 6. 【关键】将 background/neutral 样本的 Deltas 强行设为 -1e8
    # 因为回归损失只在正样本上计算，这些 -1e8 的标记会在算 Loss 时被过滤掉
    deltas[invalid_mask] = -1e8
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def rcnn_apply_deltas_to_anchors(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """
    Implement the inverse of `rcnn_get_deltas_from_anchors` here.

    Args:
        deltas: Tensor of shape `(N, 4)` giving box regression deltas.
        anchors: Tensor of shape `(N, 4)` giving anchors to apply deltas on.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving the resulting boxes in
            XYXY format.
    """

    # Clamp dw and dh such that they would transform a 8px box no larger than
    # 224px. This is necessary for numerical stability as we apply exponential.
    scale_clamp = math.log(224 / 8)
    deltas[:, 2] = torch.clamp(deltas[:, 2], max=scale_clamp)
    deltas[:, 3] = torch.clamp(deltas[:, 3], max=scale_clamp)

    ##########################################################################
    # TODO: Implement the transformation logic to get output boxes.          #
    ##########################################################################
    output_boxes = None
    # Replace "pass" statement with your code
    # 1. 拆解网络预测的 Deltas
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # 2. 将传入的 Anchor 从 XYXY 转换为 CXCYWH 格式
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0] + w_a / 2.0
    y_a = anchors[:, 1] + h_a / 2.0

    # 3. 逆向应用公式，解算出预测框的 CXCYWH
    x_pred = dx * w_a + x_a
    y_pred = dy * h_a + y_a
    w_pred = torch.exp(dw) * w_a
    h_pred = torch.exp(dh) * h_a

    # 4. 将预测框从 CXCYWH 重新转回目标要求的 XYXY 格式
    x1 = x_pred - w_pred / 2.0
    y1 = y_pred - h_pred / 2.0
    x2 = x_pred + w_pred / 2.0
    y2 = y_pred + h_pred / 2.0

    # 5. 沿着特征维度进行堆叠，形成 (N, 4) 的输出张量
    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return output_boxes


@torch.no_grad()
def sample_rpn_training(
    gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float
):
    """
    Return `num_samples` (or fewer, if not enough found) random pairs of anchors
    and GT boxes without exceeding `fg_fraction * num_samples` positives, and
    then try to fill the remaining slots with background anchors. We will ignore
    "neutral" anchors in this sampling as they are not used for training.

    Args:
        gt_boxes: Tensor of shape `(N, 5)` giving GT box co-ordinates that are
            already matched with some anchor boxes (with GT class label at last
            dimension). Label -1 means background and -1e8 means meutral.
        num_samples: Total anchor-GT pairs with label >= -1 to return.
        fg_fraction: The number of subsampled labels with values >= 0 is
            `min(num_foreground, int(fg_fraction * num_samples))`. In other
            words, if there are not enough fg, the sample is filled with
            (duplicate) bg.

    Returns:
        fg_idx, bg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or
            fewer. Use these to index anchors, GT boxes, and model predictions.
    """
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)

    # Protect against not enough foreground examples.
    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg

    # Randomly select positive and negative examples.
    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]

    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return fg_idx, bg_idx


@torch.no_grad()
def mix_gt_with_proposals(
    proposals_per_fpn_level: Dict[str, List[torch.Tensor]], gt_boxes: torch.Tensor
):
    """
    At start of training, RPN proposals may be low quality. It's possible that
    very few of these have high IoU with GT boxes. This may stall or de-stabilize
    training of second stage. This function mixes GT boxes with RPN proposals to
    improve training. Different GT boxes are mixed with proposals from different
    FPN levels according to assignment rule of FPN paper.

    Args:
        proposals_per_fpn_level: Dict of proposals per FPN level, per image in
            batch. These are same as outputs from `RPN.forward()` method.
        gt_boxes: Tensor of shape `(B, M, 4 or 5)` giving GT boxes per image in
            batch (with or without GT class label, doesn't matter).

    Returns:
        proposals_per_fpn_level: Same as input, but with GT boxes mixed in them.
    """

    # Mix ground-truth boxes for every example, per FPN level. There's no direct
    # way to vectorize this.
    for _idx, _gtb in enumerate(gt_boxes):

        # Filter empty GT boxes:
        _gtb = _gtb[_gtb[:, 4] != -1]
        if len(_gtb) == 0:
            continue

        # Compute FPN level assignments for each GT box. This follows Equation (1)
        # of FPN paper (k0 = 5). `level_assn` has `(M, )` integers, one of {3,4,5}
        _gt_area = (_gtb[:, 2] - _gtb[:, 0]) * (_gtb[:, 3] - _gtb[:, 1])
        level_assn = torch.floor(5 + torch.log2(torch.sqrt(_gt_area) / 224))
        level_assn = torch.clamp(level_assn, min=3, max=5).to(torch.int64)

        for level_name, _props in proposals_per_fpn_level.items():
            _prop = _props[_idx]

            # Get GT boxes of this image that match level scale, and append them
            # to proposals.
            _gt_boxes_fpn_subset = _gtb[level_assn == int(level_name[1])]
            if len(_gt_boxes_fpn_subset) > 0:
                proposals_per_fpn_level[level_name][_idx] = torch.cat(
                    # Remove class label since proposals don't have it:
                    [_prop, _gt_boxes_fpn_subset[:, :4]],
                    dim=0,
                )

    return proposals_per_fpn_level


class RPN(nn.Module):
    """
    Region Proposal Network: First stage of Faster R-CNN detector.

    This class puts together everything you implemented so far. It accepts FPN
    features as input and uses `RPNPredictionNetwork` to predict objectness and
    box reg deltas. Computes proposal boxes for second stage (during both
    training and inference) and losses during training.
    """

    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        """
        Args:
            batch_size_per_image: Anchors per image to sample for training.
            nms_thresh: IoU threshold for NMS - unlike FCOS, this is used
                during both, training and inference.
            pre_nms_topk: Number of top-K proposals to select before applying
                NMS, per FPN level. This helps in speeding up NMS.
            post_nms_topk: Number of top-K proposals to select after applying
                NMS, per FPN level. NMS is obviously going to be class-agnostic.

        Refer explanations of remaining args in the classes/functions above.
        """
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        # Record all input arguments:
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        ######################################################################
        # TODO: Implement the training forward pass. Follow these steps:
        #   1. Pass the FPN features per level to the RPN prediction network.
        #   2. Generate anchor boxes for all FPN levels.
        #
        # HINT: You have already implemented everything, just have to call the
        # appropriate functions.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_obj_logits, pred_boxreg_deltas, anchors_per_fpn_level = (
            None,
            None,
            None,
        )
        # Replace "pass" statement with your code
        # 1. 前向传播：把 FPN 特征送入预测网络，得到 Objectness 分数和 Box 偏移量
        # pred_obj_logits: Dict["p3/p4/p5", Tensor(B, HWA)]
        # pred_boxreg_deltas: Dict["p3/p4/p5", Tensor(B, HWA, 4)]
        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)

        # 2. 生成 Anchor：
        # 2.1 提取每层特征图的 shape (B, C, H, W)，用于计算中心点坐标
        shape_per_fpn_level = {
            level_name: feat.shape for level_name, feat in feats_per_fpn_level.items()
        }
        
        # 2.2 计算各层特征图映射回原图的中心点坐标 (x_c, y_c)
        # 注意要传入 device 确保生成的 Tensor 和 特征图在同一个 GPU 上
        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level, 
            strides_per_fpn_level,
            device=feats_per_fpn_level["p3"].device
        )
        
        # 2.3 在中心点上撒网，生成真实的 Anchor 坐标 (x1, y1, x2, y2)
        anchors_per_fpn_level = generate_fpn_anchors(
            locations_per_fpn_level,
            strides_per_fpn_level,
            stride_scale=self.anchor_stride_scale,
            aspect_ratios=self.anchor_aspect_ratios
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # We will fill three values in this output dict - "proposals",
        # "loss_rpn_box" (training only), "loss_rpn_obj" (training only)
        output_dict = {}

        # Get image height and width according to feature sizes and strides.
        # We need these to clamp proposals (These should be (224, 224) but we
        # avoid hard-coding them).
        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        # STUDENT: Implement this method before moving forward with the rest
        # of this `forward` method.
        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        # Return here during inference - loss computation not required.
        if not self.training:
            return output_dict

        # ... otherwise continue loss computation:
        ######################################################################
        # Match the generated anchors with provided GT boxes. This
        # function is not batched so you may use a for-loop, like FCOS.
        ######################################################################
        # Combine anchor boxes from all FPN levels - we do not need any
        # distinction of boxes across different levels (for training).
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        # Get matched GT boxes (list of B tensors, each of shape `(H*W*A, 5)`
        # giving matching GT boxes to anchor boxes). Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        
        # 遍历每一张图片
        for i in range(num_images):
            # 获取当前这张图匹配好的 GT boxes (包含前景、背景和忽略样本的标记)
            matched_per_image = rcnn_match_anchors_to_gt(
                anchor_boxes, 
                gt_boxes[i], 
                self.anchor_iou_thresholds
            )
            matched_gt_boxes.append(matched_per_image)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine matched boxes from all images to a `(B, HWA, 5)` tensor.
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        # Combine predictions across all FPN levels.
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            # Repeat anchor boxes `batch_size` times so there is a 1:1
            # correspondence with GT boxes.
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            # Collapse `batch_size`, and `HWA` to a single dimension so we have
            # simple `(-1, 4 or 5)` tensors. This simplifies loss computation.
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            ##################################################################
            # TODO: Compute training losses. Follow three steps in order:
            #   1. Sample a few anchor boxes for training. Pass the variable
            #      `matched_gt_boxes` to `sample_rpn_training` function and
            #      use those indices to get subset of predictions and targets.
            #      RPN samples 50-50% foreground/background anchors, unless
            #      there aren't enough foreground anchors.
            #
            #   2. Compute GT targets for box regression (you have implemented
            #      the transformation function already).
            #
            #   3. Calculate objectness and box reg losses per sampled anchor.
            #      Remember to set box loss for "background" anchors to 0.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            loss_obj, loss_box = None, None
            # Replace "pass" statement with your code
            import torch.nn.functional as F

            # Step 1: 抽样 (Sampling)
            # 计算总共需要抽样的数量 (每张图的样本数 * 图片总数)
            total_samples = self.batch_size_per_image * num_images
            # 使用框架提供的函数进行正负样本平衡抽样，返回选中框的索引
            fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, total_samples, fg_fraction=0.5)
            sample_idxs = torch.cat([fg_idx, bg_idx], dim=0)
            # 根据抽样索引，提取对应的网络预测值、Anchor坐标和匹配好的GT真值
            sampled_pred_obj = pred_obj_logits[sample_idxs]
            sampled_pred_box = pred_boxreg_deltas[sample_idxs]
            sampled_anchors = anchor_boxes[sample_idxs]
            sampled_gt_boxes = matched_gt_boxes[sample_idxs]

            # Step 2: 制作框回归的“标准答案” (Targets)
            # 利用之前写的函数，计算从 sampled_anchors 到 sampled_gt_boxes 的理论 Deltas
            target_deltas = rcnn_get_deltas_from_anchors(sampled_anchors, sampled_gt_boxes)

            # Step 3: 计算 Loss
            # 区分前景和背景：在 matched_gt_boxes 中，背景框的类别 (第5列) 被设为了 -1
            # 而 sample_rpn_training 不会返回中立样本(-1e8)，所以只要 >= 0 就是前景
            fg_mask = sampled_gt_boxes[:, 4] >= 0

            # 3.1 Objectness Loss (分类损失)
            # 前景 target 为 1.0，背景 target 为 0.0
            gt_obj = fg_mask.float()
            # 注意 reduction="none" 保证每个样本单独算损失，形状保持为 [total_samples]
            loss_obj = F.binary_cross_entropy_with_logits(
                sampled_pred_obj, gt_obj, reduction="none"
            )

            # 3.2 Box Regression Loss (回归损失)
            # L1 Loss：绝对值误差。因为预测的是 dx, dy, dw, dh，所以是四维的，在维度1求和
            loss_box = F.l1_loss(sampled_pred_box, target_deltas, reduction="none")
            loss_box = loss_box.sum(dim=1) 
            
            # 【极其关键的一步】：非前景 (背景) 的框不需要做位置回归，将其损失强行置为 0
            loss_box[~fg_mask] = 0.0
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            # Sum losses and average by num(foreground + background) anchors.
            # In training code, we simply add these two and call `.backward()`
            total_batch_size = self.batch_size_per_image * num_images
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    @torch.no_grad()  # Don't track gradients in this function.
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],  # (width, height)
    ):
        """
        Predict proposals for a batch of images for the second stage. Other
        input arguments are same as those computed in `forward` method. This
        method should not be called from anywhere except from inside `forward`.

        Returns:
            torch.Tensor
                proposals: Tensor of shape `(keep_topk, 4)` giving *absolute*
                    XYXY co-ordinates of predicted proposals. These will serve
                    as anchor boxes for the second stage.
        """

        # Gather proposals from all FPN levels in this list.
        proposals_all_levels = {
            level_name: None for level_name, _ in anchors_per_fpn_level.items()
        }
        for level_name in anchors_per_fpn_level.keys():

            # Get anchor boxes and predictions from a single level.
            level_anchors = anchors_per_fpn_level[level_name]

            # shape: (batch_size, HWA), (batch_size, HWA, 4)
            level_obj_logits = pred_obj_logits[level_name]
            level_boxreg_deltas = pred_boxreg_deltas[level_name]

            # Fill proposals per image, for this FPN level, in this list.
            level_proposals_per_image = []
            for _batch_idx in range(level_obj_logits.shape[0]):
                ##############################################################
                # TODO: Perform the following steps in order:
                #   1. Transform the anchors to proposal boxes using predicted
                #      box deltas, clamp to image height and width.
                #   2. Sort all proposals by their predicted objectness, and
                #      retain `self.pre_nms_topk` proposals. This speeds up
                #      our NMS computation. HINT: `torch.topk`
                #   3. Apply NMS and retain `keep_topk_per_level` proposals
                #      per image, per level.
                #
                # NOTE: Your `nms` method may be slow for training - you may
                # use `torchvision.ops.nms` with exact same input arguments,
                # to speed up training. We will grade your `nms` implementation
                # separately; you will NOT lose points if you don't use it here.
                #
                # Note that deltas, anchor boxes, and objectness logits have
                # different shapes, you need to make some intermediate views.
                ##############################################################
                # Replace "pass" statement with your code
                # 获取当前这张图片、在当前 FPN 层的 objectness 和 deltas
                # 形状分别为 (HWA,) 和 (HWA, 4)
                logits_per_img = level_obj_logits[_batch_idx]
                deltas_per_img = level_boxreg_deltas[_batch_idx]
                
                # --- Step 1: 变形与裁剪 ---
                # 利用之前的逆向解码函数，算出预测框的实际坐标
                proposals = rcnn_apply_deltas_to_anchors(deltas_per_img, level_anchors)
                
                # 限制坐标不要跑出图片的宽 (img_w) 和高 (img_h)
                img_w, img_h = image_size
                proposals[:, 0] = proposals[:, 0].clamp(min=0, max=img_w)
                proposals[:, 1] = proposals[:, 1].clamp(min=0, max=img_h)
                proposals[:, 2] = proposals[:, 2].clamp(min=0, max=img_w)
                proposals[:, 3] = proposals[:, 3].clamp(min=0, max=img_h)
                
                # --- Step 2: Pre-NMS Top-K 预筛选 ---
                # 确定我们要保留多少个。如果这一层的总框数还不到 pre_nms_topk，就取总框数
                num_pre_nms = min(self.pre_nms_topk, logits_per_img.shape[0])
                
                # torch.topk 极其方便，直接返回最大的 values 和对应的 indices (默认降序)
                topk_logits, topk_idx = torch.topk(logits_per_img, num_pre_nms)
                topk_proposals = proposals[topk_idx]
                
                # --- Step 3: NMS 与 Post-NMS Top-K 后筛选 ---
                # 使用 PyTorch 官方的 NMS (C++ 层面优化，比纯 Python 快得多)
                keep_idx = torchvision.ops.nms(
                    topk_proposals, topk_logits, self.nms_thresh
                )
                
                # NMS 返回的是保留下来的框的索引。我们再切片保留前 post_nms_topk 个
                keep_idx = keep_idx[:self.post_nms_topk]
                final_proposals = topk_proposals[keep_idx]
                
                # 把这张图片、这个 FPN 层的最终高价值 Proposal 加入列表
                level_proposals_per_image.append(final_proposals)
                ##############################################################
                #                        END OF YOUR CODE                    #
                ##############################################################

            # Collate proposals from individual images. Do not stack these
            # tensors, they may have different shapes since few images or
            # levels may have less than `post_nms_topk` proposals. We could
            # pad these tensors but there's no point - they will be used by
            # `torchvision.ops.roi_align` in second stage which operates
            # with lists, not batched tensors.
            proposals_all_levels[level_name] = level_proposals_per_image

        return proposals_all_levels

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)


class FasterRCNN(nn.Module):
    """
    Faster R-CNN detector: this module combines backbone, RPN, ROI predictors.

    Unlike Faster R-CNN, we will use class-agnostic box regression and Focal
    Loss for classification. We opted for this design choice for you to re-use
    a lot of concepts that you already implemented in FCOS - choosing one loss
    over other matters less overall.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        stem_channels: List[int],
        num_classes: int,
        batch_size_per_image: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules using `stem_channels` argument, exactly like
        # `FCOSPredictionNetwork` and `RPNPredictionNetwork`. use the same
        # stride, padding, and weight initialization as previous TODOs.
        #
        # HINT: This stem will be applied on RoI-aligned FPN features. You can
        # decide the number of input channels accordingly.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        cls_pred = []
        # Replace "pass" statement with your code
        cls_pred = []
        
        # 在这门课（EECS 498/598）的标准设定中，FPN 输出的特征通道数通常固定为 256。
        # 如果你的代码架构中 backbone 暴露了该属性，也可以替换为 self.backbone.out_channels
        in_channels = self.backbone.out_channels
        
        # 1. 遍历 stem_channels 构建卷积层
        for curr_channels in stem_channels:
            conv = nn.Conv2d(in_channels, curr_channels, kernel_size=3, stride=1, padding=1)
            
            # 必须的初始化：权重正态分布，偏置为 0
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.0)
            
            cls_pred.append(conv)
            cls_pred.append(nn.ReLU())
            
            # 当前层的输出通道，作为下一层的输入通道
            in_channels = curr_channels

        ######################################################################
        # TODO: Add an `nn.Flatten` module to `cls_pred`, followed by a linear
        # layer to output C+1 classification logits (C classes + background).
        # Think about the input size of this linear layer based on the output
        # shape from `nn.Flatten` layer.
        ######################################################################
        # Replace "pass" statement with your code
        # 2. 展平特征图
        cls_pred.append(nn.Flatten())
        
        # 3. 计算全连接层的输入维度
        # 经过 stride=1, padding=1 的卷积后，特征图长宽保持不变，仍为 roi_size (默认 7x7)
        # 所以展平后的向量长度 = 最后的通道数 * 长 * 宽
        flatten_dim = in_channels * self.roi_size[0] * self.roi_size[1]
        
        # 4. 构建全连接层并初始化
        # 输出维度为 num_classes (前景类别) + 1 (背景类别)
        linear = nn.Linear(flatten_dim, self.num_classes + 1)
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(linear.bias, 0.0)
        
        cls_pred.append(linear)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Wrap the layers defined by student into a `nn.Sequential` module,
        # Faster R-CNN also predicts box offsets to "refine" RPN proposals, we
        # exclude it for simplicity and keep RPN proposal boxes as final boxes.
        self.cls_pred = nn.Sequential(*cls_pred)

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        See documentation of `FCOS.forward` for more details.
        """

        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(
            feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes
        )
        proposals_per_fpn_level = output_dict["proposals"]

        # Mix GT boxes with proposals. This is necessary to stabilize training
        # since RPN proposals may be bad during first few iterations. Also, why
        # waste good supervisory signal from GT boxes, for second-stage?
        if self.training:
            proposals_per_fpn_level = mix_gt_with_proposals(
                proposals_per_fpn_level, gt_boxes
            )

        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        # Perform RoI-align using FPN features and proposal boxes.
        roi_feats_per_fpn_level = {
            level_name: None for level_name in feats_per_fpn_level.keys()
        }
        # Get RPN proposals from all levels.
        for level_name in feats_per_fpn_level.keys():
            ##################################################################
            # TODO: Call `torchvision.ops.roi_align`. See its documentation to
            # properly format input arguments. Use `aligned=True`
            ##################################################################
            level_feats = feats_per_fpn_level[level_name]
            level_props = output_dict["proposals"][level_name]
            level_stride = self.backbone.fpn_strides[level_name]

            # Replace "pass" statement with your code
            spatial_scale = 1.0 / level_stride
            
            # 调用 PyTorch 官方的 roi_align
            # 注意：level_props 已经是一个 List[Tensor] 了，正好符合官方 API 的要求
            roi_feats = torchvision.ops.roi_align(
                input=level_feats,
                boxes=level_props,
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                aligned=True
            )
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            roi_feats_per_fpn_level[level_name] = roi_feats

        # Combine ROI feats across FPN levels, do the same with proposals.
        # shape: (batch_size * total_proposals, fpn_channels, roi_h, roi_w)
        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)

        # Obtain classification logits for all ROI features.
        # shape: (batch_size * total_proposals, num_classes)
        pred_cls_logits = self.cls_pred(roi_feats)

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass. Batch size must be 1!
            # fmt: off
            return self.inference(
                images,
                proposals_per_fpn_level,
                pred_cls_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # Match the RPN proposals with provided GT boxes and append to
        # `matched_gt_boxes`. Use `rcnn_match_anchors_to_gt` with IoU threshold
        # such that IoU > 0.5 is foreground, otherwise background.
        # There are no neutral proposals in second-stage.
        ######################################################################
        matched_gt_boxes = [] 
        
        for level_name in feats_per_fpn_level.keys(): # 默认顺序: p3, p4, p5
            for _idx in range(len(gt_boxes)):         # 遍历 Batch 中的每一张图
                # 提取当前图、当前 FPN 层的 proposals
                level_props = output_dict["proposals"][level_name][_idx]
                gt_boxes_per_image = gt_boxes[_idx]
                
                # 对当前层的 proposals 进行 GT 匹配
                matched_gt = rcnn_match_anchors_to_gt(
                    level_props, 
                    gt_boxes_per_image, 
                    iou_thresholds=(0.5, 0.5)
                )
                matched_gt_boxes.append(matched_gt)
       ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine predictions and GT from across all FPN levels.
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

        ######################################################################
        # TODO: Train the classifier head. Perform these steps in order:
        #   1. Sample a few RPN proposals, like you sampled 50-50% anchor boxes
        #      to train RPN objectness classifier. However this time, sample
        #      such that ~25% RPN proposals are foreground, and the rest are
        #      background. Faster R-CNN performed such weighted sampling to
        #      deal with class imbalance, before Focal Loss was published.
        #
        #   2. Use these indices to get GT class labels from `matched_gt_boxes`
        #      and obtain the corresponding logits predicted by classifier.
        #
        #   3. Compute cross entropy loss - use `F.cross_entropy`, see its API
        #      documentation on PyTorch website. Since background ID = -1, you
        #      may shift class labels by +1 such that background ID = 0 and
        #      other VC classes have IDs (1-20). Make sure to reverse shift
        #      this during inference, so that model predicts VOC IDs (0-19).
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls = None
        # Replace "pass" statement with your code
        import torch.nn.functional as F
        
        # 1. 抽样：总共抽取 batch_size_per_image * num_images 个样本，前景占比 25% (0.25)
        total_samples = self.batch_size_per_image * num_images
        fg_idx, bg_idx = sample_rpn_training(
            matched_gt_boxes, total_samples, fg_fraction=0.25
        )
        
        # 将前景和背景的索引拼接到一起
        sample_idxs = torch.cat([fg_idx, bg_idx], dim=0)
        
        # 2. 提取抽样得到的数据
        sampled_logits = pred_cls_logits[sample_idxs]
        sampled_gt_boxes = matched_gt_boxes[sample_idxs]
        
        # 提取真实类别标签 (第5列，索引为4)
        gt_classes = sampled_gt_boxes[:, 4]
        
        # 【关键移位】：让背景类 (-1) 变成 0，让前景类别 (0~19) 变成 (1~20)
        target_classes = gt_classes + 1
        
        # 3. 计算交叉熵损失
        # 注意 PyTorch 的 cross_entropy 要求 Target 为 int64 (long) 数据类型
        loss_cls = F.cross_entropy(sampled_logits, target_classes.long())
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return {
            "loss_rpn_obj": output_dict["loss_rpn_obj"],
            "loss_rpn_box": output_dict["loss_rpn_box"],
            "loss_cls": loss_cls,
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        proposals: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        test_score_thresh: float,
        test_nms_thresh: float,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions.
        """

        # The second stage inference in Faster R-CNN is quite straightforward:
        # combine proposals from all FPN levels and perform a *class-specific
        # NMS*. There would have been more steps here if we further refined
        # RPN proposals by predicting box regression deltas.

        # Use `[0]` to remove the batch dimension.
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)

        ######################################################################
        # Faster R-CNN inference, perform the following steps in order:
        #   1. Get the most confident predicted class and score for every box.
        #      Note that the "score" of any class (including background) is its
        #      probability after applying C+1 softmax.
        #
        #   2. Only retain prediction that have a confidence score higher than
        #      provided threshold in arguments.
        #
        # NOTE: `pred_classes` may contain background as ID = 0 (based on how
        # the classifier was supervised in `forward`). Remember to shift the
        # predicted IDs such that model outputs ID (0-19) for 20 VOC classes.
        ######################################################################
        pred_scores, pred_classes = None, None
        # Replace "pass" statement with your code
        import torch.nn.functional as F

        # 1. 将预测的 Logits 转换为概率 (概率总和为 1)
        # pred_cls_logits 的形状是 (N, C + 1)
        probs = F.softmax(pred_cls_logits, dim=-1)

        # 取出每个框概率最大的那个类别，以及对应的概率分数
        # pred_scores: (N,) 对应的最高分数
        # pred_classes: (N,) 对应的类别索引
        pred_scores, pred_classes = torch.max(probs, dim=-1)

        # 2. 设置过滤掩码 (Mask)：
        # 条件 A: 类别必须大于 0 (因为我们在训练时给所有类别加了 1，0 成了背景)
        # 条件 B: 置信度分数必须大于我们设定的阈值 test_score_thresh
        keep_mask = (pred_classes > 0) & (pred_scores > test_score_thresh)

        # 利用掩码过滤掉不合格的框和分数
        pred_boxes = pred_boxes[keep_mask]
        pred_scores = pred_scores[keep_mask]
        pred_classes = pred_classes[keep_mask]

        # 3. 类别 ID 复位：
        # 将保留下来的前景类别 (1~20) 整体减 1，恢复为外部评测接口所期望的 VOC 标签 (0~19)
        pred_classes = pred_classes - 1
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        # STUDENTS: This line depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh
        )
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_classes, pred_scores
