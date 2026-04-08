import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for a
    tiny RegNet model so it can train decently with a single K80 Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next
    # hidden state and any values you need for the backward pass in the next_h
    # and cache variables respectively.
    ##########################################################################
    # Replace "pass" statement with your code
    next_h = torch.tanh(x.mm(Wx) + prev_h.mm(Wh) + b)
    
    # 5. 将反向传播(Backward)需要用到的变量存入 cache
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms of the output value from tanh.
    ##########################################################################
    # Replace "pass" statement with your code
    x, prev_h, Wx, Wh, b, next_h = cache
    
    # 1. 反向穿过 tanh 函数 (计算局部梯度)
    # 对应公式: da = dnext_h * (1 - tanh^2(a))
    da = dnext_h * (1 - next_h ** 2)
    
    # 2. 反向穿过矩阵乘法 (注意维度对齐和转置 .T)
    # dx: (N, H) @ (H, D) -> (N, D)
    dx = da.mm(Wx.T)
    # dWx: (D, N) @ (N, H) -> (D, H)
    dWx = x.T.mm(da)
    
    # dprev_h: (N, H) @ (H, H) -> (N, H)
    dprev_h = da.mm(Wh.T)
    # dWh: (H, N) @ (N, H) -> (H, H)
    dWh = prev_h.T.mm(da)
    
    # 3. 反向穿过加法偏置 (在 Batch 维度 N 上求和，保持形状为 (H,))
    db = da.sum(dim=0)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Args:
        x: Input data for the entire timeseries, of shape (N, T, D).
        h0: Initial hidden state, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        h: Hidden states for the entire timeseries, of shape (N, T, H).
        cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of
    # input data. You should use the rnn_step_forward function that you defined
    # above. You can use a for loop to help compute the forward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    # 获取维度信息
    N, T, D = x.shape
    # 从权重矩阵推断出隐藏层维度 H
    H = Wh.shape[0] 
    
    # 1. 准备“日记本”：初始化用于保存所有时间步 hidden state 的张量
    # x.new_zeros 是一种优雅的写法，可以确保 h 和 x 拥有相同的数据类型和设备(CPU/GPU)
    h = x.new_zeros((N, T, H))
    
    # 2. 准备列表用于保存所有时间步的 cache
    cache = []
    
    # 初始化当前的隐藏状态，最初等于 h0
    curr_h = h0
    
    # 3. 沿时间维度展开循环
    for t in range(T):
        # 提取当前时间步的输入，形状为 (N, D)
        x_t = x[:, t, :]
        
        # 调用你之前写好的单步前向传播函数
        next_h, step_cache = rnn_step_forward(x_t, curr_h, Wx, Wh, b)
        
        # 将结果写入“日记本”的对应时间步
        h[:, t, :] = next_h
        
        # 更新当前隐藏状态，为下一步做准备
        curr_h = next_h
        
        # 存下这一步的 cache
        cache.append(step_cache)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Args:
        dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
        dx: Gradient of inputs, of shape (N, T, D)
        dh0: Gradient of initial hidden state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire
    # sequence of data. You should use the rnn_step_backward function that you
    # defined above. You can use a for loop to help compute the backward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    # 从 cache 的第一个元素中提取出我们需要推断维度的信息
    x_first, prev_h_first, Wx_first, Wh_first, b_first, next_h_first = cache[0]
    N, D = x_first.shape
    H = Wh_first.shape[0]
    T = len(cache)
    
    # 1. 初始化所有需要返回的梯度矩阵
    dx = dh.new_zeros((N, T, D))
    dWx = dh.new_zeros((D, H))
    dWh = dh.new_zeros((H, H))
    db = dh.new_zeros((H,))
    
    # 2. 初始化从“未来”传回来的隐藏状态梯度，一开始全是 0
    dh_prev = dh.new_zeros((N, H))
    
    # 3. 时间倒流：从 T-1 步一路退回到 0 步
    for t in reversed(range(T)):
        # 取出这一步对应的前向传播记录
        step_cache = cache[t]
        
        # 核心逻辑：当前步的总梯度 = 外部传来的梯度 dh[:, t, :] + 未来传回的梯度 dh_prev
        dh_total = dh[:, t, :] + dh_prev
        
        # 调用单步反向传播函数
        dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dh_total, step_cache)
        
        # 4. 将计算结果分配或累加
        # 输入 x 的梯度是按时间步独立存放的
        dx[:, t, :] = dx_t
        
        # 权重 Wx, Wh, b 在所有时间步是共享的，必须将梯度累加起来！
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
        
        # 将当前步算出的前一个隐藏状态的梯度，传递给下一次循环（即 t-1 步）
        dh_prev = dprev_h_t
        
    # 当整个循环结束时，dh_prev 里装的就是对初始隐藏状态 h0 的梯度
    dh0 = dh_prev
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
            Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
            Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
            b: Biases, of shape (H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Args:
        x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
        out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):

        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.
        ######################################################################
        # Replace "pass" statement with your code
        out = self.W_embed[x]
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Args:
        x: Input scores, of shape (N, T, V)
        y: Ground-truth indices, of shape (N, T) where each element is in the
            range 0 <= y[i, t] < V

    Returns a tuple of:
        loss: Scalar giving loss
    """
    loss = None

    ##########################################################################
    # TODO: Implement the temporal softmax loss function.
    #
    # REQUIREMENT: This part MUST be done in one single line of code!
    #
    # HINT: Look up the function torch.functional.cross_entropy, set
    # ignore_index to the variable ignore_index (i.e., index of NULL) and
    # set reduction to either 'sum' or 'mean' (avoid using 'none' for now).
    #
    # We use a cross-entropy loss at each timestep, *summing* the loss over
    # all timesteps and *averaging* across the minibatch.
    ##########################################################################
    # Replace "pass" statement with your code
    loss = F.cross_entropy(x.transpose(1, 2), y, ignore_index=ignore_index, reduction='sum') / x.shape[0]
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
            word_to_idx: A dictionary giving the vocabulary. It contains V
                entries, and maps each string to a unique integer in the
                range [0, V).
            input_dim: Dimension D of input image feature vectors.
            wordvec_dim: Dimension W of word vectors.
            hidden_dim: Dimension H for the hidden state of the RNN.
            cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        ######################################################################
        # TODO: Initialize the image captioning module. Refer to the TODO
        # in the captioning_forward function on layers you need to create
        #
        # You may want to check the following pre-defined classes:
        # ImageEncoder WordEmbedding, RNN, LSTM, AttentionLSTM, nn.Linear
        #
        # (1) output projection (from RNN hidden state to vocab probability)
        # (2) feature projection (from CNN pooled feature to h0)
        ######################################################################
        # Replace "pass" statement with your code
        self.project_features = nn.Linear(input_dim, hidden_dim)
        
        # 2. 词嵌入层：将词汇索引映射为 W 维的词向量
        self.embed = WordEmbedding(vocab_size, wordvec_dim)
        
        # 3. 核心序列生成器：根据指定的网络类型实例化
        # 它的输入是词向量 (W维)，隐藏状态是 (H维)
        if self.cell_type == "rnn":
            self.rnn = RNN(wordvec_dim, hidden_dim)
        elif self.cell_type == "lstm":
            self.rnn = LSTM(wordvec_dim, hidden_dim)
        elif self.cell_type == "attn":
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim)
            
        # 4. 输出投影层：将 H 维的隐藏状态映射为 V 维的词汇表概率得分
        self.project_output = nn.Linear(hidden_dim, vocab_size)
        
        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss. The
        backward part will be done by torch.autograd.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            captions: Ground-truth captions; an integer array of shape (N, T + 1)
                where each element is in the range 0 <= y[i, t] < V

        Returns:
            loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last
        # word and will be input to the RNN; captions_out has everything but the
        # first word and this is what we will expect the RNN to generate. These
        # are offset by one relative to each other because the RNN should produce
        # word (t+1) after receiving word t. The first element of captions_in
        # will be the START token, and the first element of captions_out will
        # be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.
        # In the forward pass you will need to do the following:
        # (1) Use an affine transformation to project the image feature to
        #     the initial hidden state $h0$ (for RNN/LSTM, of shape (N, H)) or
        #     the projected CNN activation input $A$ (for Attention LSTM,
        #     of shape (N, H, 4, 4).
        # (2) Use a word embedding layer to transform the words in captions_in
        #     from indices to vectors, giving an array of shape (N, T, W).
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to
        #     process the sequence of input word vectors and produce hidden state
        #     vectors for all timesteps, producing an array of shape (N, T, H).
        # (4) Use a (temporal) affine transformation to compute scores over the
        #     vocabulary at every timestep using the hidden states, giving an
        #     array of shape (N, T, V).
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring
        #     the points where the output word is <NULL>.
        #
        # Do not worry about regularizing the weights or their gradients!
        ######################################################################
        # Replace "pass" statement with your code
        # 1. 看图：提取图像的卷积特征 (形状为 N, D, 4, 4)
        features = self.image_encoder(images) 
        
        if self.cell_type == 'attn':
            # Attention 模型需要保留空间结构
            # 因为 project_features 是一个普通的 Linear 层，它作用在张量的最后一个维度上
            # 所以我们需要把维度 (N, D, 4, 4) 互换为 (N, 4, 4, D)，经过 Linear 变成 (N, 4, 4, H)
            # 最后再换回 AttentionLSTM 期望的 (N, H, 4, 4)
            features_perm = features.permute(0, 2, 3, 1)
            A = self.project_features(features_perm)
            A = A.permute(0, 3, 1, 2)
            
            # 2. 查字典：获取输入句子的词向量 (N, T, W)
            word_embeds = self.embed(captions_in)
            
            # 3. 思考：运行 AttentionLSTM
            h = self.rnn(word_embeds, A)
            
        else:
            # 普通 RNN / LSTM 模型不需要空间结构，直接用均值池化 (Mean Pooling) 压扁
            features_pooled = features.mean(dim=(2, 3)) # (N, D)
            h0 = self.project_features(features_pooled) # 得到初始隐藏状态 h0 (N, H)
            
            # 2. 查字典：获取输入句子的词向量 (N, T, W)
            word_embeds = self.embed(captions_in)
            
            # 3. 思考：运行普通 RNN 或 LSTM
            h = self.rnn(word_embeds, h0)
            
        # 4. 说话：将每个时间步的隐藏状态映射为词汇表得分 (N, T, V)
        scores = self.project_output(h)
        
        # 5. 判卷：计算时间序列上的 Softmax 交叉熵损失
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self.ignore_index)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            max_length: Maximum length T of generated captions

        Returns:
            captions: Array of shape (N, max_length) giving sampled captions,
                where each element is an integer in the range [0, V). The first
                element of captions should be the first sampled word, not the
                <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        ######################################################################
        # TODO: Implement test-time sampling for the model. You will need to
        # initialize the hidden state of the RNN by applying the learned affine
        # transform to the image features. The first word that you feed to
        # the RNN should be the <START> token; its value is stored in the
        # variable self._start. At each timestep you will need to do to:
        # (1) Embed the previous word using the learned word embeddings
        # (2) Make an RNN step using the previous hidden state and the embedded
        #     current word to get the next hidden state.
        # (3) Apply the learned affine transformation to the next hidden state to
        #     get scores for all words in the vocabulary
        # (4) Select the word with the highest score as the next word, writing it
        #     (the word index) to the appropriate slot in the captions variable
        #
        # For simplicity, you do not need to stop generating after an <END> token
        # is sampled, but you can if you want to.
        #
        # NOTE: we are still working over minibatches in this function. Also if
        # you are using an LSTM, initialize the first cell state to zeros.
        # For AttentionLSTM, first project the 1280x4x4 CNN feature activation
        # to $A$ of shape Hx4x4. The LSTM initial hidden state and cell state
        # would both be A.mean(dim=(2, 3)).
        #######################################################################
        # Replace "pass" statement with your code
        features = self.image_encoder(images)
        
        if self.cell_type == 'attn':
            # 注意力模型：保留 4x4 的空间结构
            features_perm = features.permute(0, 2, 3, 1)
            A = self.project_features(features_perm)
            A = A.permute(0, 3, 1, 2)
            
            # 初始记忆和细胞状态是图像特征的平均值
            h = A.mean(dim=(2, 3))
            c = h
        else:
            # 普通 RNN / LSTM 模型：直接压扁
            features_pooled = features.mean(dim=(2, 3))
            h = self.project_features(features_pooled)
            
            if self.cell_type == 'lstm':
                # LSTM 的初始细胞状态设为全零
                c = torch.zeros_like(h)

        # 2. 准备第一个输入词：对于 Batch 中的每张图片，第一个词都是 <START> 标记
        current_word = images.new(N).fill_(self._start).long()
        
        # 3. 开始自回归循环：逐个单词生成句子
        for t in range(max_length):
            # 将当前词索引转换为词向量，形状 (N, W)
            word_embed = self.embed(current_word)
            
            # 运行模型的一个时间步 (Step Forward)
            if self.cell_type == 'rnn':
                h = self.rnn.step_forward(word_embed, h)
            
            elif self.cell_type == 'lstm':
                h, c = self.rnn.step_forward(word_embed, h, c)
                
            elif self.cell_type == 'attn':
                # 注意力机制特有步骤：先计算当前的隐藏状态应该关注图片的哪个区域
                attn, attn_weights = dot_product_attention(h, A)
                # 将关注点、词向量一起喂给 LSTM
                h, c = self.rnn.step_forward(word_embed, h, c, attn)
                # 保存这一步的注意力权重，方便后续做热力图可视化
                attn_weights_all[:, t, :, :] = attn_weights
                
            # 将更新后的隐藏状态映射为整个词汇表的得分，形状 (N, V)
            scores = self.project_output(h)
            
            # 找出得分最高的那个词的索引！
            # max(dim=1) 会返回两个值：最高分数本身(我们不需要) 和 对应的索引(我们需要)
            _, next_word = scores.max(dim=1)
            
            # 把预测出来的词填入最终的答题卷
            captions[:, t] = next_word
            
            # 极其重要的一步：将预测出的词作为下一个时间步的输入！
            current_word = next_word
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    """Single-layer, uni-directional LSTM module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            Wx: Input-to-hidden weights, of shape (D, 4H)
            Wh: Hidden-to-hidden weights, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                next_h: Next hidden state, of shape (N, H)
                next_c: Next cell state, of shape (N, H)
        """
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.
        ######################################################################
        next_h, next_c = None, None
        # Replace "pass" statement with your code
        # 获取隐藏层维度 H (因为 Wx 的形状是 (D, 4H)，所以除以 4)
        H = self.Wx.shape[1] // 4
        
        # 第 1 步：大一统的线性变换，算出所有门的生数据 (Logits)
        # 形状变化：A 的形状将是 (N, 4H)
        A = x.mm(self.Wx) + prev_h.mm(self.Wh) + self.b
        
        # 第 2 步：沿着第二个维度（列）将矩阵切分成 4 等份，每份 (N, H)
        # 顺序通常约定俗成为：输入门 i, 遗忘门 f, 输出门 o, 候选记忆 g
        A_i = A[:, 0:H]
        A_f = A[:, H:2*H]
        A_o = A[:, 2*H:3*H]
        A_g = A[:, 3*H:4*H]
        
        # 第 3 步：套用激活函数，将生数据变成真正的门控值和候选记忆
        i = torch.sigmoid(A_i)
        f = torch.sigmoid(A_f)
        o = torch.sigmoid(A_o)
        g = torch.tanh(A_g)
        
        # 第 4 步：更新长期日记本 (细胞状态 c)
        # 注意：这里的 * 是逐元素相乘 (Element-wise multiplication)
        next_c = f * prev_c + i * g
        
        # 第 5 步：计算当前时刻要对外的短期记忆 (隐藏状态 h)
        next_h = o * torch.tanh(next_c)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not returned;
        it is an internal variable to the LSTM and is not accessed from outside.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output.
        """

        c0 = torch.zeros_like(
            h0
        )  # we provide the intial cell state c0 here for you!
        ######################################################################
        # TODO: Implement the forward pass for an LSTM over entire timeseries
        ######################################################################
        hn = None
        # Replace "pass" statement with your code
        # 获取维度信息
        N, T, D = x.shape
        H = h0.shape[1]
        
        # 准备一个空日记本，记录所有时间步的 h (最终需要返回的输出)
        # 使用 x.new_zeros 保持数据类型和设备(CPU/GPU)一致
        hn = x.new_zeros((N, T, H))
        
        # 初始化当前时刻的 h 和 c
        curr_h = h0
        curr_c = c0
        
        # 沿着时间维度 T 展开循环
        for t in range(T):
            # 获取当前时刻的输入单词向量，形状 (N, D)
            x_t = x[:, t, :]
            
            # 喂给单步车间，得到更新后的 h 和 c
            next_h, next_c = self.step_forward(x_t, curr_h, curr_c)
            
            # 把当前时刻算出的 h 记录到总输出张量里
            hn[:, t, :] = next_h
            
            # 击鼓传花：把现在的状态变成“上一步”的状态，传给下一次循环
            curr_h = next_h
            curr_c = next_c
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return hn


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
        prev_h: The LSTM hidden state from previous time step, of shape (N, H)
        A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Returns:
        attn: Attention embedding output, of shape (N, H)
        attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention we described earlier. #
    # You will use this function for `AttentionLSTM` forward and sample      #
    # functions. HINT: Make sure you reshape attn_weights back to (N, 4, 4)! #
    ##########################################################################
    # Replace "pass" statement with your code
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention we described earlier. 
    ##########################################################################
    
    # 1. 展平 CNN 图纸：把 4x4 的网格铺开变成 16
    # 形状：(N, H, 4, 4) -> (N, H, 16)
    A_flat = A.view(N, H, D_a * D_a)
    
    # 2. 计算点积得分：Q (prev_h) 乘以 K (A_flat)
    # prev_h 扩维：(N, H) -> (N, 1, H)
    # 矩阵乘法：(N, 1, H) @ (N, H, 16) -> (N, 1, 16) -> 降维得到 (N, 16)
    scores = torch.bmm(prev_h.unsqueeze(1), A_flat).squeeze(1)
    
    # 3. 缩放 (Scale)：除以 sqrt(H) 防止梯度消失/爆炸
    scores = scores / math.sqrt(H)
    
    # 4. Softmax 概率归一化：算出每个格子分配多少注意力
    # 形状：(N, 16)
    attn_weights_flat = F.softmax(scores, dim=1)
    
    # 5. 加权求和，获得最终的 Context Vector (attn)
    # A_flat 作为 V：(N, H, 16)
    # 权重扩维：(N, 16) -> (N, 16, 1)
    # 矩阵乘法：(N, H, 16) @ (N, 16, 1) -> (N, H, 1) -> 降维得到 (N, H)
    attn = torch.bmm(A_flat, attn_weights_flat.unsqueeze(2)).squeeze(2)
    
    # 6. 还原注意力权重的形状，方便后续画出 4x4 的可视化热力图
    # 形状：(N, 16) -> (N, 4, 4)
    attn_weights = attn_weights_flat.view(N, D_a, D_a)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
        input_dim: Input size, denoted as D before
        hidden_dim: Hidden size, denoted as H before
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            attn: The attention embedding, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
            next_c: The next cell state, of shape (N, H)
        """

        #######################################################################
        # TODO: Implement forward pass for a single timestep of attention LSTM.
        # Feel free to re-use some of your code from `LSTM.step_forward()`.
        #######################################################################
        next_h, next_c = None, None
        # Replace "pass" statement with your code
        # 从 Wx 推断隐藏层维度 H (Wx 形状是 D x 4H)
        H = self.Wx.shape[1] // 4
        
        # 1. 核心变化：将输入词 x、历史隐藏状态 prev_h、视觉上下文 attn 全部投影并相加
        # 此时 A 的形状为 (N, 4H)
        A = x.mm(self.Wx) + prev_h.mm(self.Wh) + attn.mm(self.Wattn) + self.b
        
        # 2. 沿列切分出四个部分，分别对应输入门(i)、遗忘门(f)、输出门(o)和候选记忆(g)
        A_i = A[:, 0:H]
        A_f = A[:, H:2*H]
        A_o = A[:, 2*H:3*H]
        A_g = A[:, 3*H:4*H]
        
        # 3. 套用激活函数 (与普通 LSTM 完全一致)
        i = torch.sigmoid(A_i)
        f = torch.sigmoid(A_f)
        o = torch.sigmoid(A_o)
        g = torch.tanh(A_g)
        
        # 4. 更新细胞状态 (长期记忆)
        next_c = f * prev_c + i * g
        
        # 5. 计算当前时刻的隐藏状态 (短期记忆)
        next_h = o * torch.tanh(next_c)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM uses
        a hidden size of H, and we work over a minibatch containing N sequences.
        After running the LSTM forward, we return hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it
        is an internal variable to the LSTM and is not accessed from outside.

        h0 and c0 are same initialized as the global image feature (meanpooled A)
        For simplicity, we implement scaled dot-product attention, which means in
        Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
        f_{att}(a_i, h_{t-1}) equals to the scaled dot product of a_i and h_{t-1}.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Returns:
            hn: The hidden state output
        """

        # The initial hidden state h0 and cell state c0 are initialized
        # differently in AttentionLSTM from the original LSTM and hence
        # we provided them for you.
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)

        ######################################################################
        # TODO: Implement the forward pass for an LSTM over an entire time-  #
        # series. You should use the `dot_product_attention` function that   #
        # is defined outside this module.                                    #
        ######################################################################
        hn = None
        # Replace "pass" statement with your code
        N, T, D = x.shape
        H = h0.shape[1]
        
        # 1. 准备“日记本”，记录每个时间步的输出隐藏状态
        hn = x.new_zeros((N, T, H))
        
        # 2. 初始化当前状态
        curr_h = h0
        curr_c = c0
        
        # 3. 沿时间维度展开循环
        for t in range(T):
            # 取出当前时刻的输入词向量，形状 (N, D)
            x_t = x[:, t, :]
            
            # --- 注意力核心逻辑 ---
            # 第 1 步：拿着“上一步”的隐藏状态 curr_h 当 Query，去图片 A (Key) 中做匹配
            # 返回加权后的视觉上下文 attn，以及注意力权重 attn_weights (这里前向传播不需要返回权重)
            attn, _ = dot_product_attention(curr_h, A)
            
            # 第 2 步：拿着刚找到的视觉重点 attn，以及当前单词 x_t、上一步状态，送入 LSTM 细胞
            next_h, next_c = self.step_forward(x_t, curr_h, curr_c, attn)
            
            # 记录这一步的输出
            hn[:, t, :] = next_h
            
            # 状态滚动：击鼓传花给下一个时间步
            curr_h = next_h
            curr_c = next_c
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return hn
