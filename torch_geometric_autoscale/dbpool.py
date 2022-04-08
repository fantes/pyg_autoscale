from typing import Optional, Callable

import torch
from torch import Tensor
from torch.cuda import Stream

db_synchronize = torch.ops.torch_geometric_autoscale.db_synchronize
db_read_async = torch.ops.torch_geometric_autoscale.db_read_async
db_write_async = torch.ops.torch_geometric_autoscale.db_write_async


class DBAsyncIOPool(torch.nn.Module):
    def __init__(self, pool_size: int, buffer_size: int, embedding_dim: int):
        super().__init__()

        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.embedding_dim = embedding_dim

        self._device = torch.device("cpu")
        self._pull_queue = []
        self._push_cache = [None] * pool_size
        # self._push_streams = [None] * pool_size
        self._pull_streams = [None] * pool_size
        self._cuda_buffers = [None] * pool_size
        self._pull_index = -1
        self._push_index = -1

    def _apply(self, fn: Callable) -> None:
        self._device = fn(torch.zeros(1)).device
        return self

    def _pull_stream(self, idx: int) -> Stream:
        if self._pull_streams[idx] is None:
            assert str(self._device)[:4] == "cuda"
            self._pull_streams[idx] = torch.cuda.Stream(self._device)
        return self._pull_streams[idx]

    # def _push_stream(self, idx: int) -> Stream:
    #     if self._push_streams[idx] is None:
    #         assert str(self._device)[:4] == "cuda"
    #         self._push_streams[idx] = torch.cuda.Stream(self._device)
    #     return self._push_streams[idx]

    def _cuda_buffer(self, idx: int) -> Tensor:
        if self._cuda_buffers[idx] is None:
            assert str(self._device)[:4] == "cuda"
            self._cuda_buffers[idx] = torch.empty(
                self.buffer_size, self.embedding_dim, device=self._device
            )
        return self._cuda_buffers[idx]

    @torch.no_grad()
    def async_pull_from_db(
        self,
        layer: int,
        offset: Optional[Tensor],
        count: Optional[Tensor],
        index: Tensor,
    ) -> None:
        # Start pulling `src` at ([offset, count] and index positions:
        self._pull_index = (self._pull_index + 1) % self.pool_size
        data = (self._pull_index, layer, offset, count, index)
        print("adding to queue pull from db of layer", layer)
        self._pull_queue.append(data)
        if len(self._pull_queue) <= self.pool_size:
            self._async_pull_from_db(self._pull_index, layer, offset, count, index)

    @torch.no_grad()
    def _async_pull_from_db(
        self,
        idx: int,
        layer: int,
        offset: Optional[Tensor],
        count: Optional[Tensor],
        index: Tensor,
    ) -> None:
        print("launching async read db->cuda of layer ", layer)
        with torch.cuda.stream(self._pull_stream(idx)):
            db_read_async(layer, offset, count, index, self._cuda_buffer(idx))

    @torch.no_grad()
    def synchronize_pull_from_db(self) -> Tensor:
        # Synchronize the next pull command:
        print("sync pulling layer", self._pull_queue[0][1])
        idx = self._pull_queue[0][0]
        print("sync thread")
        db_synchronize()
        print("sync stream")
        torch.cuda.synchronize(self._pull_stream(idx))
        print("returning cuda buffer of layer", self._pull_queue[0][1])
        return self._cuda_buffer(idx)

    @torch.no_grad()
    def free_pull_from_db(self) -> None:
        # Free the buffer space and start pulling from remaining queue:
        print("freeing pull request")
        self._pull_queue.pop(0)
        if len(self._pull_queue) >= self.pool_size:
            data = self._pull_queue[self.pool_size - 1]
            idx, layer, offset, count, index = data
            self._async_pull_from_db(idx, layer, offset, count, index)
        elif len(self._pull_queue) == 0:
            self._pull_index = -1

    @torch.no_grad()
    def async_push_to_db(
        self, layer: int, src: Tensor, offset: Tensor, count: Tensor
    ) -> None:
        # Start pushing `src` to ([offset, count] and index positions to `dst`:
        print("lauchning async push to db")
        self._push_index = (self._push_index + 1) % self.pool_size
        print("sync previous push idx", self._push_index)
        self.synchronize_push_to_db(self._push_index)
        self._push_cache[self._push_index] = src
        # with torch.cuda.stream(self._push_stream(self._push_index)):
        print("lauchning write async of layer ", layer)
        db_write_async(layer, src, offset, count)

    @torch.no_grad()
    def synchronize_push_to_db(self, idx: Optional[int] = None) -> None:
        # Synchronize the push command of stream `idx` or all commands:
        if idx is None:
            for idx in range(self.pool_size):
                self.synchronize_push_to_db(idx)
            self._push_index = -1
        else:
            print("sync thread for push idx", idx)
            db_synchronize()
            # torch.cuda.synchronize(self._push_stream(idx))
            self._push_cache[idx] = None

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(pool_size={self.pool_size}, "
            f"buffer_size={self.buffer_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self._device})"
        )
