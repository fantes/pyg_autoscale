import importlib
import os.path as osp

import torch

__version__ = '0.0.0'

for library in ['_relabel', '_async']:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

from .data import get_data  # noqa
from .history import History  # noqa
from .dbhistory import DBHistory  # noqa
from .pool import AsyncIOPool  # noqa
from .metis import metis, permute  # noqa
from .utils import compute_micro_f1, gen_masks, dropout  # noqa
from .loader import SubgraphLoader, EvalSubgraphLoader  # noqa
from .models import ScalableGNN, DBScalableGNN

__all__ = [
    'get_data',
    'History',
    'DBHistory',
    'AsyncIOPool',
    'metis',
    'permute',
    'compute_micro_f1',
    'gen_masks',
    'dropout',
    'SubgraphLoader',
    'EvalSubgraphLoader',
    'ScalableGNN',
    'DBScalableGNN',
    '__version__',
]
