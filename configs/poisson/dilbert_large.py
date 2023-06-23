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

# Lint as: python3
from configs.default_mnist_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'poisson'
  training.continuous = True
  training.batch_size = 64
  training.small_batch_size = 32
  training.gamma = 5
  training.restrict_M = True
  training.tau = 0.03
  training.snapshot_freq = 500
  training.model = 'ddpmpp'
  training.M = 356
  training.beta = 0.5
  training.grace_period = 2500


  # data
  data = config.data
  data.channels = 3
  data.centered = False
  data.dataset = 'dilbert_large'
  data.image_size = 512
  data.classes = 6
  data.random_flip = True


  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.ode_solver = 'rk45'

  sampling.N = 100
  sampling.z_max = 100
  sampling.z_min = 1e-3
  sampling.upper_norm = 30000
  # verbose
  sampling.vs = False
  sampling.target = 0
  sampling.snr = 0.075
  sampling.N = 1000
  sampling.z_exp = 0.1

  # model
  # model
  model = config.model

  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'

  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 4, 4)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.embedding_type = 'positional'
  model.conv_size = 3
  model.sigma_end = 0.01
  return config
