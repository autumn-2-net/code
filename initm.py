import torch
from torch import nn
import torchexplorer

from hifivae import HiFivae
cfg={'upsample_rates': [16, 16, 2], 'upsample_kernel_sizes': [32, 32, 4], 'upsample_initial_channel': 256, 'Eupsample_initial_channel': 256, 'resblock_kernel_sizes': [3, 7, 11], 'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 'Eresblock_kernel_sizes': [15], 'Eresblock_dilation_sizes': [[1, 3, 5]], 'discriminator_periods': [3, 5, 7, 11, 17, 23, 37], 'resblock': '1', 'codedim': 512, 'codenum': 256}


# print(config['model_args'])
mmmx=HiFivae(cfg)
model = mmmx
dummy_X = torch.randn(20, 1,8192)
torchexplorer.watch(model, log_freq=1, backend='standalone')
# model.training_step({'audio':dummy_X},1)['loss'].backward()
model(dummy_X).sum().backward()