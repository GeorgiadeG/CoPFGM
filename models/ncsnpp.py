# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
    # all_resolutions.insert(0, config.data.image_size+1)

    self.num_channels = config.data.num_channels
    self.num_classes = config.data.classes
    self.img_size = config.data.image_size


    # n = config.training.small_batch_size * config.data.num_channels * config.data.classes
    # self.conv1 = nn.Conv2d(in_channels=config.data.num_channels,out_channels=config.data.num_channels
    #                             ,kernel_size=(1, config.data.classes+1))
    # self.conv1 = nn.Conv2d(config.data.num_channels + config.data.classes, config.data.num_channels, 1)

    # A Conv2D layer with a kernel size of (1, num_classes+1) to transform the concatenated tensor back to original width
    # self.conv1 = nn.Conv2d(num_channels, num_channels, (1, num_classes + 1), stride=0, pad)

    # m = config.eval.batch_size * config.data.num_channels * config.data.classes
    # self.conv1_train = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=m+1,stride=1)
    # self.conv1 = nn.Conv1d(1, 1, kernel_size=num_classes + 1, stride=num_classes + 1, bias=False)
    # self.fc1 = nn.Linear(num_channels * img_height * img_width + num_classes * num_channels,
    #                      num_channels * img_height * img_width)

    # self.linear = nn.Linear(self.img_size ** 2 + self.num_classes, self.img_size ** 2)

    # self.combine = Combine(self.num_channels, self.num_classes, method='cat')
    # self.downsample = conv1x1(self.num_channels + self.num_classes, self.num_channels)

    # For sparl_cc
    # self.conv1 = nn.Conv2d(self.num_channels + self.num_classes, self.num_channels, kernel_size=1)

    # For sparl_ewad
    # self.proj = nn.Conv2d(self.num_classes, self.num_channels, kernel_size=1)

    # For class conditional batch norm
    # self.fc = nn.Linear(self.num_classes, self.num_classes * self.img_size * self.img_size)
    # self.conv1 = nn.Conv2d(self.num_channels + self.num_classes, self.num_channels, kernel_size=1)

    # For adain
    # self.class_embedding = ClassEmbedding(self.num_classes, self.num_channels)

    # Conditional Batch Normalization
    # self.bn1 = ConditionalBatchNorm(self.num_channels, self.num_classes)

    # self.resnet_block = ResnetBlockOneHot(act=nn.ReLU(), in_ch=self.num_channels, zemb_dim=self.num_classes)

    # Define CNN layers
    # self.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=3, padding=1)
    # self.conv2 = nn.Conv2d(64, self.num_channels, kernel_size=3, padding=1)

    # Define additional FC layer for one-hot vectors
    # self.fc_one_hot = nn.Linear(self.num_classes, self.img_size * self.img_size)

    self.small_batch_size = config.training.small_batch_size
    self.conditional = conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    self.progressive = progressive = config.model.progressive.lower()
    self.progressive_input = progressive_input = config.model.progressive_input.lower()
    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    init_scale = config.model.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.model.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # z/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.model.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      zemb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      zemb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # channels = config.data.num_channels + config.data.classes
    channels = config.data.num_channels+1
    # channels = config.data.num_channels
    if progressive_input != 'none':
      input_pyramid_ch = channels

    modules.append(conv3x3(channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      if self.config.training.sde == 'poisson':
        # output an extra channel for PFGM z-dimension
        modules.append(conv3x3(in_ch, config.data.num_channels + 1, init_scale=init_scale))
      else:
        modules.append(conv3x3(in_ch, config.data.num_channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, cond, labels=None):

    # # Ensure one_hot is in the same device as x
    # one_hot = one_hot.to(x.device)
    #
    # # Expand the dimensions of one_hot to match that of x
    # # Assuming x has shape (batch_size, channels, height, width)
    # # and one_hot has shape (batch_size, num_classes)
    # one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)
    #
    # # Repeat one_hot across the spatial dimensions
    # # so it has the same shape as x
    # one_hot = one_hot.expand(-1, -1, x.shape[2], x.shape[3])
    #
    # # Concatenate the one-hot encodings to x along the channel dimension
    # x = torch.cat([x, one_hot], dim=1)

    # batch_size, num_channels, img_height, img_width = x.size()
    #
    # # Expand dimensions of one_hot to match with input tensor x.
    # one_hot_expanded = one_hot.view(batch_size, 1, 1, -1).repeat(1, num_channels, 1, 1)
    #
    # # Now, one_hot_expanded has size [batch_size, num_channels, 1, num_classes]
    #
    # # Then, expand the dimensions to match the height of image
    # one_hot_expanded = one_hot_expanded.expand(-1, -1, img_height, -1)
    #
    # # Now, one_hot_expanded has size [batch_size, num_channels, img_height, num_classes]
    #
    # # Concatenate x and one_hot_expanded along the width dimension
    # new_x = torch.cat([x, one_hot_expanded], dim=3)
    #
    # x = self.conv1_test(new_x)

    # Reshape the input tensor to 2D: (batch_size * num_channels, img_size^2)
    # batch_size = x.shape[0]
    # x = x.reshape(batch_size * self.num_channels, -1)
    #
    # # Repeat and reshape the labels to match x's first dimension
    # one_hot = labels.repeat(1, self.num_channels).view(-1, self.num_classes)
    #
    # # Concatenate the reshaped input and one-hot labels
    # x = torch.cat([x, one_hot], dim=1)
    #
    # # Use a linear layer to scale down to (batch_size*num_channels, img_size^2)
    # x = self.linear(x)
    #
    # # Reshape it to the initial format
    # x = x.view(batch_size, self.num_channels, self.img_size, self.img_size)

    # x = self.bn1(x, labels)
    # x = F.pad(x, (0,1,0,1))

    # x = self.resnet_block(x.float(), labels.float())

    # x = self.sparl_ew(x, labels, op='mul')

    # # CNN forward pass
    # x = F.relu(self.conv1(x))
    # x = self.conv2(x)
    #
    # # One-hot forward pass
    # one_hot = F.relu(self.fc_one_hot(labels.float()))
    #
    # # Reshape one-hot to have same dimensions as feature map
    # one_hot = one_hot.view(one_hot.size(0), 1, x.size(2), x.size(3))
    #
    # # Add (or multiply) one-hot representations to feature maps
    # x = x + one_hot

    # x = self.sparl_cc(x, labels)

    extra_channel = torch.zeros(x.shape[0], 1, self.img_size, self.img_size).to(x.device)

    for i in range(x.shape[0]):
      for j in range(x.shape[2]):
        extra_channel[i, 0, j, -self.num_classes:] = labels[i]

    x = torch.cat([x, extra_channel], dim=1)

    # assert x.shape[0] == batch_size
    # assert x.shape[1] == self.num_channels
    # assert x.shape[2] == self.img_size
    # assert x.shape[3] == self.img_size

    # z (PFGM)/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      # used z (PFGM) or sigmas

      ### only used for score-based models (VE)
      used_sigmas = cond
      ###

      zemb = modules[m_idx](torch.log(cond))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.

      ### only used for score-based models (VE)
      used_sigmas = self.sigmas[cond.long()]
      ###

      zemb = layers.get_positional_embedding(cond, self.nf)
    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      zemb = modules[m_idx](zemb)
      m_idx += 1
      zemb = modules[m_idx](self.act(zemb))
      m_idx += 1
    else:
      zemb = None

    if not self.config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], zemb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], zemb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, zemb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, zemb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), zemb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, zemb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
    if self.config.model.scale_by_sigma:
      # only for score-based models VE
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    if self.config.training.sde == 'poisson':
      # Predict the direction on the extra z dimension
      scalar = F.adaptive_avg_pool2d(h[:, -1], (1, 1))
      return h[:, :-1], scalar.reshape(len(scalar))
    else:
      return h

  # Spatial Replication of Labels + Concatenation + Convolution
  def sparl_cc(self, x, labels):
    assert labels is not None, "Labels must be provided for sparl_cc"
    """
    Spatial Replication of Labels + Concatenation + Convolution
    :param x: input tensor
    :param labels: labels tensor
    :return: output tensor
    """
    # reshape labels from (batch_size, num_classes) to (batch_size, num_classes, 1, 1)
    labels = labels.view(labels.size(0), labels.size(1), 1, 1)

    # repeat labels tensor to match the spatial dimensions of x
    labels = labels.repeat(1, 1, x.size(2), x.size(3))

    # concatenate labels tensor with x along the channels dimension
    combined = torch.cat([x, labels], dim=1)

    # scale down the number of channels using the 1x1 convolution layer
    return self.conv1(combined)

  # Spatial Replication of Labels + Element-wise Addition/Multiplication:
  def sparl_ew(self, x, labels, op='add'):
    assert labels is not None, "Labels must be provided for sparl_ew"
    """
    Spatial Replication of Labels + Element-wise Addition/Multiplication
    :param x: input tensor
    :param labels: labels tensor
    :return: output tensor
    """
    # reshape labels from (batch_size, num_classes) to (batch_size, num_classes, 1, 1)
    labels = labels.view(labels.size(0), labels.size(1), 1, 1)

    # repeat labels tensor to match the spatial dimensions of x
    labels = labels.repeat(1, 1, x.size(2), x.size(3))

    # element-wise addition/multiplication
    labels_proj = self.proj(labels.float())

    if op == 'add':
        return x + labels_proj
    elif op == 'mul':
        return x * labels_proj
    else:
        raise ValueError(f'{op} is not a valid name for Spatial '
                         f'Replication of Labels + '
                         f'Element-wise Addition/Multiplication:')

  # Class-conditional Batch Normalization:
  def cc_bn(self, x, labels):
    assert labels is not None, "Labels must be provided for cc_bn"
    """
    Class-conditional Batch Normalization
    :param x: input tensor
    :param labels: labels tensor
    :return: output tensor
    """
    labels = labels.float()
    # reshape labels from (batch_size, num_classes) to (batch_size, num_channels*img_sz*img_sz)
    labels = self.fc(labels)

    # reshape labels to match the spatial dimensions of x
    labels = labels.view(-1, self.num_classes, self.img_size, self.img_size)

    x_and_labels = torch.cat([x, labels], dim=1)

    return self.conv1(x_and_labels)

  # Adaptive Instance Normalization (AdaIN):
  def adain(self, x, labels):
    assert labels is not None, "Labels must be provided for adain"
    """
    Adaptive Instance Normalization (AdaIN)
    :param x: input tensor
    :param labels: labels tensor
    :return: output tensor
    """
    labels = self.class_embedding(labels)  # shape: (batch_size, num_channels)

    # Expand labels to match the spatial dimensions of x
    labels = labels.unsqueeze(-1).unsqueeze(-1)  # shape: (batch_size, num_channels, 1, 1)
    labels = labels.expand(-1, -1, x.size(2), x.size(3))  # shape: (batch_size, num_channels, img_sz, img_sz)

    # Add the label embeddings to x along the channel dimension
    return x + labels  # shape: (batch_size, num_channels, img_sz, img_sz)

class ClassEmbedding(nn.Module):
  def __init__(self, num_classes, num_channels):
    super(ClassEmbedding, self).__init__()
    self.fc = nn.Linear(num_classes, num_channels)

  def forward(self, labels):
    # Embed the labels to the same space as channels
    labels = labels.float()
    labels = self.fc(labels)  # shape: (batch_size, num_channels)

    # The labels are now in the same space as your image tensor's channel dimension.
    # You can add them to your image tensor along the channel dimension.

    return labels

class ConditionalBatchNorm(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Linear(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
    self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    y = y.float()  # ensure y is float
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out