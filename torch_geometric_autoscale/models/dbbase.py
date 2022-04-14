from typing import Optional, Callable, Dict, Any

import warnings

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric_autoscale import DBHistory, DBAsyncIOPool, AsyncDBHistory
from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader


class DBScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs via historical
    embeddings.
    This class will take care of initializing :obj:`num_layers - 1` historical
    embeddings, and provides a convenient interface to push recent node
    embeddings to the history, and to pull previous embeddings from the
    history.
    In case historical embeddings are stored on the CPU, they will reside
    inside pinned memory, which allows for asynchronous memory transfers of
    historical embeddings.
    For this, this class maintains a :class:`AsyncIOPool` object that
    implements the underlying mechanisms of asynchronous memory transfers as
    described in our paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        hidden_channels (int): The number of hidden channels of the model.
            As a current restriction, all intermediate node embeddings need to
            utilize the same number of features.
        num_layers (int): The number of layers of the model.
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_channels: int,
        num_layers: int,
        pool_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        device=None,
        multi_db=True,
        db_value_from_tensor=None,
        tensor_from_dbvalue=None,
        list_from_dbvalue=None,
        asyncio=False,
        db_path="/tmp/dbhist.db_",
        debug_threading=False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers if pool_size is None else pool_size
        self.buffer_size = buffer_size
        self.db_path = db_path
        self.debug_threading = debug_threading

        if not asyncio:
            self.dbhistory = DBHistory(
                self.num_nodes,
                self.num_layers,
                self.hidden_channels,
                multi_db=multi_db,
                db_value_from_tensor=db_value_from_tensor,
                tensor_from_dbvalue=tensor_from_dbvalue,
                list_from_dbvalue=list_from_dbvalue,
            )
        else:
            self.dbhistory = AsyncDBHistory(
                self.num_nodes,
                self.num_layers,
                self.hidden_channels,
                db_path=self.db_path,
            )

        self.pool: Optional[DBAsyncIOPool] = None
        self._async = asyncio

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.dbhistory._device

    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        if self._async:
            self.pool = DBAsyncIOPool(
                self.pool_size, self.buffer_size, self.hidden_channels
            )
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        self.dbhistory.reset_parameters()

    def __call__(
        self,
        x: Optional[Tensor] = None,
        adj_t: Optional[SparseTensor] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        loader: EvalSubgraphLoader = None,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj:`3` and :obj:`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj:`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj:`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj:`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj:`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj:`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj:`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
        """

        if loader is not None:
            print("not doing mini inference")

        if self._async:
            if self.debug_threading:
                print("start pulling every layer", flush=True)
            for i in range(self.num_layers):
                self.pool.async_pull_from_db(i, None, None, n_id[batch_size:])

        if self.debug_threading:
            print("start forwarding", flush=True)
        out = self.forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)

        if self._async:
            if self.debug_threading:
                print("syncing all pushes")
            self.pool.synchronize_push_to_db()

        return out

    def push_and_pull(
        self,
        layer: int,
        x: Tensor,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""

        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            self.dbhistory.push(x, n_id=None, layer=layer)
            return x

        assert n_id is not None

        if batch_size is None:
            self.dbhistory.push(x, n_id, layer=layer)
            return x

        if not self._async:
            self.dbhistory.push(
                x[:batch_size],
                n_id=n_id[:batch_size],
                layer=layer,
                offset=offset,
                count=count,
            )
            h = self.dbhistory.pull(n_id=n_id[batch_size:], layer=layer)
            return torch.cat([x[:batch_size], h], dim=0)

        else:
            if self.debug_threading:
                print("launching push at layer", layer, flush=True)
            self.pool.async_push_to_db(layer, x[:batch_size], offset, count)
            if self.debug_threading:
                print("reading first pull at layer", layer, flush=True)
            out = self.pool.synchronize_pull_from_db()[: n_id.numel() - batch_size]
            out = torch.cat([x[:batch_size], out], dim=0)
            if self.debug_threading:
                print("removing pull from pool ", flush=True)
            self.pool.free_pull_from_db()
            return out

    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels, pin_memory=True)
        return self.__out

    @torch.no_grad()
    def forward_layer(
        self, layer: int, x: Tensor, adj_t: SparseTensor, state: Dict[str, Any]
    ) -> Tensor:
        raise NotImplementedError
