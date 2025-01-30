import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'

import torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("Available CUDA devices:", torch.cuda.device_count())

import os

# 打印环境变量
print(os.environ.get('HF_DATASETS_CACHE'))