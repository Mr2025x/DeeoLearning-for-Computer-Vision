"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a6_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "pass" statement with your code
    return content_weight * torch.sum((content_current - content_original) ** 2)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = features.shape
    # 将 H 和 W 维度展平
    features_flat = features.view(N, C, H * W)
    
    # 计算批量矩阵乘法: (N, C, H*W) x (N, H*W, C) -> (N, C, C)
    gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
    
    if normalize:
        gram /= (H * W * C)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss = 0.0
    # 遍历所有需要计算风格的层
    for i, layer_idx in enumerate(style_layers):
        # 取出当前层的特征，计算其 Gram 矩阵
        current_gram = gram_matrix(feats[layer_idx])
        # 计算当前层的风格损失并累加
        loss += style_weights[i] * torch.sum((current_gram - style_targets[i]) ** 2)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "pass" statement with your code
    pass
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  # Replace "pass" statement with your code
  N, R, C, H, W = features.shape
    # masks 目前是 (N, R, H, W)，需要扩展一个通道维度 (N, R, 1, H, W) 才能与特征相乘
  masks = masks.unsqueeze(2)
    
    # 将 mask 应用到特征上
  masked_features = features * masks
    
    # 为了复用批量矩阵乘法逻辑，我们把 N 和 R 合并到批次维度
  masked_features = masked_features.view(N * R, C, H * W)
    
    # 计算 Gram 矩阵
  guided_gram = torch.bmm(masked_features, masked_features.transpose(1, 2))
    
  if normalize:
        guided_gram /= (H * W * C)
        
    # 把形状恢复为 (N, R, C, C)
  return guided_gram.view(N, R, C, C)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "pass" statement with your code
    loss = 0.0
    for i, layer_idx in enumerate(style_layers):
        current_feat = feats[layer_idx]
        current_mask = content_masks[layer_idx]
        
        # 将当前特征维度扩展为 (N, 1, C, H, W) 以匹配 mask 的区域逻辑
        # 假设只有一个大区域或者这里需要根据上下文推断特征扩展方式
        N, C, H, W = current_feat.shape
        R = current_mask.shape[1]
        
        # 将特征在 R 维度上复制
        current_feat_expanded = current_feat.unsqueeze(1).expand(N, R, C, H, W)
        
        # 计算 guided gram matrix
        current_guided_gram = guided_gram_matrix(current_feat_expanded, current_mask)
        
        # 计算损失
        loss += style_weights[i] * torch.sum((current_guided_gram - style_targets[i]) ** 2)
        
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
