# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ===================== NEW: IBN-a & BlurPool2d =====================
class IBN2d(nn.Module):
    """
    IBN-a: split channels, first ratio*channels -> IN (no running stats),
           remaining -> BN (with running stats).
    """
    def __init__(self, num_features, ratio=0.25, eps=1e-5, momentum=0.1):
        super().__init__()
        split = int(num_features * ratio)
        self.split = split
        self.num_features = num_features
        self.IN = nn.InstanceNorm2d(split, affine=True, eps=eps, track_running_stats=False) if split > 0 else None
        self.BN = nn.BatchNorm2d(num_features - split, eps=eps, momentum=momentum) if num_features - split > 0 else None
        if self.BN is not None:
            self.BN.weight.data.fill_(1.0)
            self.BN.bias.data.zero_()
        if self.IN is not None:
            self.IN.weight.data.fill_(1.0)
            self.IN.bias.data.zero_()

    def forward(self, x):
        if self.split == 0:
            return self.BN(x)
        if self.split == self.num_features:
            return self.IN(x)
        x1, x2 = torch.split(x, [self.split, self.num_features - self.split], dim=1)
        y1 = self.IN(x1) if self.IN is not None else x1
        y2 = self.BN(x2) if self.BN is not None else x2
        return torch.cat([y1, y2], dim=1)


class BlurPool2d(nn.Module):
    """
    轻量抗混叠下采样：固定低通核 + stride 下采样
    默认使用 1,2,1 的分离可分核（等价于 [[1,2,1],[2,4,2],[1,2,1]]/16）
    """
    def __init__(self, channels, stride=2, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 5), "BlurPool2d only supports kernel_size 3 or 5"
        if kernel_size == 3:
            filt_1d = torch.tensor([1., 2., 1.])
        else:
            filt_1d = torch.tensor([1., 4., 6., 4., 1.])
        filt_2d = filt_1d[:, None] @ filt_1d[None, :]
        filt_2d = filt_2d / filt_2d.sum()
        self.register_buffer('kernel', filt_2d[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.pad = kernel_size // 2
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.pad, groups=self.groups)
# ==================================================================


class RefConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, map_k=3):
    super(RefConv, self).__init__()
    assert map_k <= kernel_size
    self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
    self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
    G = in_channels * out_channels // (groups ** 2)
    self.num_2d_kernels = out_channels * in_channels // groups
    self.kernel_size = kernel_size
    self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                             out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                             groups=G, bias=False)

    self.bias = None
    self.stride = stride
    self.groups = groups
    if padding is None:
      padding = kernel_size // 2
    self.padding = padding

  def forward(self, inputs):
    origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
    kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
    out = F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                   bias=self.bias)
    return out

# --- gaussian initialize ---
def init_layer(L):
  # Initialization using fan-in
  if isinstance(L, nn.Conv2d):
    n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
    L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
  elif isinstance(L, nn.BatchNorm2d):
    L.weight.data.fill_(1)
    L.bias.data.fill_(0)

class distLinear(nn.Module):
  def __init__(self, indim, outdim):
    super(distLinear, self).__init__()
    self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
    self.relu = nn.ReLU()

  def forward(self, x):
    x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
    self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
    cos_dist = self.L(x_normalized)
    scores = 10 * cos_dist
    return scores

# --- flatten tensor ---
class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)


# --- Linear module ---
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
  def __init__(self, in_features, out_features, bias=True):
    super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
    self.weight.fast = None #Lazy hack to add fast weight link
    self.bias.fast = None

  def forward(self, x):
    if self.weight.fast is not None and self.bias.fast is not None:
      out = F.linear(x, self.weight.fast, self.bias.fast)
    else:
      out = super(Linear_fw, self).forward(x)
    return out

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
    super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    self.weight.fast = None
    if not self.bias is None:
      self.bias.fast = None

  def forward(self, x):
    if self.bias is None:
      if self.weight.fast is not None:
        out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    else:
      if self.weight.fast is not None and self.bias.fast is not None:
        out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    return out

# --- softplus module ---
def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  feature_augment = False
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    if self.feature_augment: # initialize {gamma, beta} with {0.3, 0.5}
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
      self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    if self.feature_augment and self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*out + beta
    return out

# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- Simple Conv Block ---
class ConvBlock(nn.Module):
  def __init__(self, indim, outdim, pool=True, padding=1, use_refconv=True):
    super(ConvBlock, self).__init__()
    self.indim = indim
    self.outdim = outdim

    # If RefConv is enabled, use it, else fallback to standard Conv2d
    self.C = RefConv(indim, outdim, 3, padding=padding) if use_refconv else nn.Conv2d(indim, outdim, 3, padding=padding)

    self.BN = nn.BatchNorm2d(outdim)
    self.relu = nn.ReLU(inplace=True)
    self.parametrized_layers = [self.C, self.BN, self.relu]
    if pool:
      self.pool = nn.MaxPool2d(2)
      self.parametrized_layers.append(self.pool)

    for layer in self.parametrized_layers:
      init_layer(layer)

    self.trunk = nn.Sequential(*self.parametrized_layers)

  def forward(self, x):
    return self.trunk(x)


# ---------------------- EMA 模块 ----------------------
class EMA(nn.Module):
  def __init__(self, channels, factor=32):
    super(EMA, self).__init__()
    self.groups = factor
    assert channels // self.groups > 0, "通道数必须能被组数整除"
    self.softmax = nn.Softmax(-1)
    self.agp = nn.AdaptiveAvgPool2d((1, 1))
    self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
    self.pool_w = nn.AdaptiveAvgPool2d((1, None))
    self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
    self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
    self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)

  def forward(self, x):
    b, c, h, w = x.size()
    group_x = x.reshape(b * self.groups, -1, h, w)  # [B*g, C/g, H, W]

    # 1x1分支
    x_h = self.pool_h(group_x)  # [B*g, C/g, H, 1]
    x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # [B*g, C/g, W, 1]
    hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # [B*g, C/g, H+W, 1]
    x_h, x_w = torch.split(hw, [h, w], dim=2)
    x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

    # 3x3分支
    x2 = self.conv3x3(group_x)

    # 跨空间融合
    x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
    x12 = x2.reshape(b * self.groups, c // self.groups, -1)
    x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
    x22 = x1.reshape(b * self.groups, c // self.groups, -1)
    weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

    return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# --- Simple ResNet Block (BlurPool does downsampling) ---
class SimpleBlock(nn.Module):
  maml = False
  def __init__(self, indim, outdim, half_res, leaky=False):
    super(SimpleBlock, self).__init__()
    self.indim = indim
    self.outdim = outdim
    self.half_res = half_res

    # 使用 BlurPool 完成几何下采样（仅在 stage 的第一个 block）
    self.down = BlurPool2d(indim, stride=2) if half_res else nn.Identity()

    # 是否使用 IBN-a（仅 stage1/2：outdim <= 128）
    use_ibn = (outdim <= 128)

    # 主分支：几何下采样已经由 self.down 完成，这里统一 stride=1
    if self.maml:
      self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=1, padding=1, bias=False)
      self.BN1 = BatchNorm2d_fw(outdim)
      self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, stride=1, padding=1, bias=False)
      self.BN2 = FeatureWiseTransformation2d_fw(outdim)
    else:
      self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1, bias=False)
      self.BN1 = (IBN2d(outdim, ratio=0.25) if use_ibn else nn.BatchNorm2d(outdim))
      self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, padding=1, bias=False)
      self.BN2 = (IBN2d(outdim, ratio=0.25) if use_ibn else nn.BatchNorm2d(outdim))

    # 保持你的 EMA
    self.ema = EMA(outdim)

    self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
    self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

    self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

    # 捷径分支：几何下采样也已由 self.down 完成，这里统一 stride=1
    if indim != outdim:
      if self.maml:
        self.shortcut = Conv2d_fw(indim, outdim, kernel_size=1, stride=1, bias=False)
        self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
      else:
        self.shortcut = nn.Conv2d(indim, outdim, kernel_size=1, stride=1, bias=False)
        self.BNshortcut = (IBN2d(outdim, ratio=0.25) if use_ibn else nn.BatchNorm2d(outdim))
      self.parametrized_layers += [self.shortcut, self.BNshortcut]
      self.shortcut_type = '1x1'
    else:
      self.shortcut_type = 'identity'

    for layer in self.parametrized_layers:
      init_layer(layer)

  def forward(self, x):
    # 先用 BlurPool 做几何下采样（若 half_res=True）
    x_in = self.down(x)

    out = self.C1(x_in)
    out = self.BN1(out)
    out = self.relu1(out)

    out = self.C2(out)
    out = self.BN2(out)

    # EMA 在残差相加前
    out = self.ema(out)

    short_out = x_in if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x_in))
    out = out + short_out
    out = self.relu2(out)
    return out



# --- ConvNet module ---
class ConvNet(nn.Module):
  def __init__(self, depth, flatten = True):
    super(ConvNet,self).__init__()
    self.grads = []
    self.fmaps = []
    trunk = []
    for i in range(depth):
      indim = 3 if i == 0 else 64
      outdim = 64
      B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
      trunk.append(B)

    if flatten:
      trunk.append(Flatten())

    self.trunk = nn.Sequential(*trunk)
    self.final_feat_dim = 1600

  def forward(self,x):
    out = self.trunk(x)
    return out

# --- ConvNetNopool module ---
class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
  def __init__(self, depth):
    super(ConvNetNopool,self).__init__()
    self.grads = []
    self.fmaps = []
    trunk = []
    for i in range(depth):
      indim = 3 if i == 0 else 64
      outdim = 64
      B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
      trunk.append(B)

    self.trunk = nn.Sequential(*trunk)
    self.final_feat_dim = [64,19,19]

  def forward(self,x):
    out = self.trunk(x)
    return out

# --- ResNet module ---
#---------------------- 修改后的ResNet - ---------------------


class ResNet(nn.Module):

  def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False):
    super(ResNet, self).__init__()
    assert len(list_of_num_layers) == 4, "需要4个stage的参数"

    # 初始层
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # 残差阶段（关键修改点）
    self.stage1 = self._make_stage(block, 64, list_of_out_dims[0], list_of_num_layers[0], leakyrelu)
    self.stage2 = self._make_stage(block, list_of_out_dims[0], list_of_out_dims[1], list_of_num_layers[1], leakyrelu)
    self.stage3 = self._make_stage(block, list_of_out_dims[1], list_of_out_dims[2], list_of_num_layers[2], leakyrelu)
    self.stage4 = self._make_stage(block, list_of_out_dims[2], list_of_out_dims[3], list_of_num_layers[3], leakyrelu)

    # 分类头
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if flatten else nn.Identity()
    self.flatten = Flatten() if flatten else nn.Identity()

    # Add the final feature dimension
    self.final_feat_dim = list_of_out_dims[3]  # The number of output channels in the last stage

  def _make_stage(self, block, in_dim, out_dim, num_blocks, leakyrelu):
    layers = []
    # 每个stage的第一个block可能需要下采样
    layers.append(block(in_dim, out_dim, half_res=(in_dim != out_dim), leaky=leakyrelu))
    # 后续blocks保持相同维度
    for _ in range(1, num_blocks):
      layers.append(block(out_dim, out_dim, half_res=False, leaky=leakyrelu))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.stage1(x)  # [B, 64, 56, 56]
    x = self.stage2(x)  # [B, 128, 28, 28]
    x = self.stage3(x)  # [B, 256, 14, 14]
    x = self.stage4(x)  # [B, 512, 7, 7]

    x = self.avgpool(x)  # [B, 512, 1, 1]
    return self.flatten(x)  # [B, 512]


# --- Conv networks ---
def Conv4():
    return ConvNet(4)
def Conv6():
    return ConvNet(6)
def Conv4NP():
    return ConvNetNopool(4)
def Conv6NP():
    return ConvNetNopool(6)

# --- ResNet networks ---
def ResNet10_EMA(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, leakyrelu)
def ResNet18_EMA(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, leakyrelu)
def ResNet34(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten, leakyrelu)

model_dict = dict(Conv4 = Conv4,
                  Conv6 = Conv6,
                  ResNet10_EMA=ResNet10_EMA,
                  ResNet18_EMA=ResNet18_EMA,
                  ResNet34 = ResNet34)

if __name__ == "__main__":
    model = ResNet10_EMA()
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)  # 正确输出 torch.Size([2, 512])
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
