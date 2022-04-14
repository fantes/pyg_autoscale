from typing import Optional, Union, List

import torch
import rocksdb
import io
import os
import shutil
from torch import Tensor

read_async = torch.ops.torch_geometric_autoscale.db_read_async
write_async = torch.ops.torch_geometric_autoscale.db_write_async
init_db = torch.ops.torch_geometric_autoscale.init_db
delete_db = torch.ops.torch_geometric_autoscale.delete_db


class AsyncDBHistory(torch.nn.Module):
    r"""A historical embedding storage module, using c++ db."""
    # num_embeddings is num_nodes, embedding_dim is hidden_dim
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        embedding_dim: int,
        device=None,
        db_path="/tmp/dbhist.db",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self._device = torch.device("cpu")
        self._dbpath = db_path
        self.reset_parameters()
        # init_db(self._dbpath, self.num_layers)

    def reset_parameters(self):
        # print("calling delete_db", flush=True)
        delete_db()
        # print("deleting dirs", flush=True)
        if os.path.exists(self._dbpath):
            shutil.rmtree(self._dbpath)
        # print("calling init_db", flush=True)
        init_db(self._dbpath, self.num_layers)

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def push(
        self,
        x,
        n_id: Tensor = None,
        layer: int = None,
    ):

        x = x.to(torch.device("cpu"))

        if n_id is None:
            db_write_async(layer, x, torch.Tensor([0]), torch.Tensor([x.size(0)]))
        else:
            assert n_id.size(0) == x.size(0)
            db_write_async(layer, x, n_id, torch.Tensor([1] * x.size(0)))

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError
