"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    # 1. 前向传播，获取所有类别的得分
    scores = model(X)
    
    # 2. 提取每个样本对应的正确类别的得分
    # X.shape[0] 是 batch size N
    correct_scores = scores[torch.arange(X.shape[0]), y]
    
    # 3. 将得分求和作为标量损失，以便我们可以对其调用 backward()
    # 注意：这里我们是为了最大化得分，但 backward() 默认计算梯度，所以直接求和即可
    loss = correct_scores.sum()
    loss.backward()
    
    # 4. 获取输入图像的梯度，取绝对值，并在通道维度 (dim=1) 上取最大值
    # X.grad.data 的形状是 (N, 3, H, W)
    # torch.max 返回 (values, indices)，我们只需要 values，所以用 [0] 或解包
    saliency, _ = torch.max(X.grad.data.abs(), dim=1)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # Replace "pass" statement with your code
    for i in range(max_iter):
        # 前向传播
        scores = model(X_adv)
        
        # 检查是否已经成功欺骗模型（预测类别等于目标类别）
        if scores[0].argmax() == target_y:
            if verbose:
                print(f"Attack succeeded in {i} iterations.")
            break
            
        # 获取目标类别的得分
        target_score = scores[0, target_y]
        
        # 反向传播计算关于 X_adv 的梯度
        target_score.backward()
        
        # 提取梯度并进行归一化
        grad = X_adv.grad.data
        grad_norm = torch.norm(grad, p=2)
        
        # 如果范数不为0，则进行梯度上升步更新 (In-place operation)
        if grad_norm > 0:
            dX = learning_rate * grad / grad_norm
            X_adv.data += dX
            
        # 极其重要：在下一次迭代前，必须将梯度清零！
        X_adv.grad.zero_()
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    # 1. 前向传播
    scores = model(img)
    target_score = scores[0, target_y]
    
    # 2. 反向传播计算梯度
    target_score.backward()
    
    # 3. 提取梯度
    grad = img.grad.data
    
    # 4. 执行梯度上升，并加入 L2 正则化的惩罚项
    # 注意：由于是梯度上升，我们要加上得分的梯度，减去正则化的梯度
    img.data += learning_rate * (grad - 2 * l2_reg * img.data)
    
    # 5. 清空梯度，为下一步做准备
    img.grad.zero_()
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
