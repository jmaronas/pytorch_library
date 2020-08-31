#-*- coding: utf-8 -*-
# config.py : Config file to be used by all the code
# Author: Juan Maroñas 
import torch
import math
import platform


def check_device():
        device = 'cpu'
        if torch.cuda.is_available():
                device = 'cuda'
        return device

def check_torch(torch_version):
        if torch.__version__ != torch_version:
                raise ImportError('Torch does not match correct version {}'.format(torch_version))

## Config Variables
torch_version = '1.5.0'
device=check_device()
dtype = torch.float32
torch.set_default_dtype(dtype)
is_linux = 'linux' in platform.platform().lower()

## Constant definitions
pi = torch.tensor(math.pi,dtype=dtype).to(device)
epsilon = torch.tensor(1e-11)

## Callers
check_torch(torch_version)
