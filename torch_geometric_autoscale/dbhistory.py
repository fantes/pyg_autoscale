from typing import Optional, Union, List

import torch
import rocksdb
import io
import os
import shutil
from torch import Tensor


class DBHistory(torch.nn.Module):
    r"""A historical embedding storage module, using db."""
    # num_embeddings is num_nodes, embedding_dim is hidden_dim
    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        embedding_dim: int,
        multi_db=False,
        device=None,
        db_value_from_tensor=None,
        tensor_from_dbvalue=None,
        list_from_dbvalue=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.multi_db = multi_db
        if db_value_from_tensor is None:
            self.db_value_from_tensor = self.db_value_from_tensor_torch_save
        else:
            self.db_value_from_tensor = db_value_from_tensor
        if tensor_from_dbvalue is None:
            self.tensor_from_dbvalue = self.tensor_from_dbvalue_torch_load
        else:
            self.tensor_from_dbvalue = tensor_from_dbvalue
        if list_from_dbvalue is None:
            self.list_from_dbvalue = self.list_from_dbvalue_torch_load
        else:
            self.list_from_dbvalue = list_from_dbvalue

        if self.multi_db:
            self.db = []
            self.opts = []
            for j in range(self.num_layers):
                self.opts.append(rocksdb.Options())
                self.opts[j].create_if_missing = True
                self.opts[j].error_if_exists = True
                self.opts[j].max_open_files = 300000
                # self.opts[j].write_buffer_size = 6710886400
                self.opts[j].max_write_buffer_number = 2
                # self.opts[j].target_file_size_base = 67108864
                self.opts[j].table_factory = rocksdb.BlockBasedTableFactory(
                    filter_policy=rocksdb.BloomFilterPolicy(10),
                    # block_cache=rocksdb.LRUCache(20 * (1024 ** 3)),
                    # block_cache_compressed=rocksdb.LRUCache(5000 * (1024 ** 2)))
                )

                if os.path.exists("/tmp/graphhist_" + str(j) + ".db"):
                    shutil.rmtree("/tmp/graphhist_" + str(j) + ".db")
                db = rocksdb.DB("/tmp/graphhist_" + str(j) + ".db", self.opts[j])
                self.db.append(db)

        else:
            self.opts = rocksdb.Options()
            self.opts.create_if_missing = True
            self.opts.error_if_exists = True
            self.opts.max_open_files = 300000
            # self.opts.write_buffer_size = 6710886400
            self.opts.max_write_buffer_number = 2
            # self.opts.target_file_size_base = 67108864
            self.opts.table_factory = rocksdb.BlockBasedTableFactory(
                filter_policy=rocksdb.BloomFilterPolicy(10),
                # block_cache=rocksdb.LRUCache(20 * (1024 ** 3)),
                # block_cache_compressed=rocksdb.LRUCache(5000 * (1024 ** 2)))
            )

            if os.path.exists("/tmp/graphhist.db"):
                shutil.rmtree("/tmp/graphhist.db")
            self.db = rocksdb.DB("/tmp/graphhist.db", self.opts)

        self.wbatch = rocksdb.WriteBatch()
        self._device = torch.device("cpu")

    def key(self, num_node, layer):
        if self.multi_db:
            return bytes(str(num_node), "utf-8")
        return bytes(str(num_node) + "." + str(layer), "utf-8")

    def db_value_from_tensor_torch_save(self, v):
        assert v.dim() == 1
        assert v.shape[0] == self.embedding_dim
        buf = io.BytesIO()
        torch.save(v, buf)
        return buf.getvalue()

    def tensor_from_dbvalue_torch_load(self, v, hidden_dim):
        buf = io.BytesIO(v)
        return torch.load(buf)

    def list_from_dbvalue_torch_load(self, v, hidden_dim):
        buf = io.BytesIO(v)
        return torch.load(buf).tolist()

    def reset_parameters(self):
        if self.multi_db:
            self.db = []
            for j in range(self.num_layers):
                if os.path.exists("/tmp/graphhist_" + str(j) + ".db"):
                    shutil.rmtree("/tmp/graphhist_" + str(j) + ".db")
                db = rocksdb.DB("/tmp/graphhist_" + str(j) + ".db", self.opts[j])
                self.db.append(db)

        else:
            if os.path.exists("/tmp/graphhist.db"):
                shutil.rmtree("/tmp/graphhist.db")
            self.db = rocksdb.DB("/tmp/graphhist.db", self.opts)
        # if self.multi_db:
        #     for j in range(self.num_layers):
        #         for i in range(self.num_nodes):
        #             self.wbatch.delete(self.key(i,j))
        #         self.db[j].write(self.wbatch)
        #         self.wbatch.clear()
        # else:
        #     for i in range(self.num_nodes):
        #         for j in range(self.num_layers):
        #             self.wbatch.delete(self.key(i,j))
        #     self.db.write(self.wbatch)
        #     self.wbatch.clear()

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, n_id: Tensor = None, layer: int = None) -> Tensor:
        keys = [self.key(i.item(), layer) for i in n_id]
        if self.multi_db:
            vv = self.db[layer].multi_get(keys)
        else:
            vv = self.db.multi_get(keys)
        # out = torch.stack([self.tensor_from_dbvalue(vv[k],self.embedding_dim) for k in keys]).pin_memory()
        out = torch.Tensor(
            [self.list_from_dbvalue(vv[k], self.embedding_dim) for k in keys]
        )

        if out.shape[0] == 1:
            out.squeeze_(0)
        return out.to(device=self._device, non_blocking=True)

    @torch.no_grad()
    def push(
        self,
        x,
        n_id: Tensor = None,
        layer: int = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
    ):
        # print("start push")
        x = x.to(torch.device("cpu"))

        if offset is None or count is None:
            # print("start push w/o offset,id")
            if n_id.shape[0] == 1:
                if self.multi_db:
                    self.db[layer].put(
                        self.key(n_id.item(), layer),
                        self.db_value_from_tensor(x),
                        disable_wal=True,
                    )
                else:
                    self.db.put(
                        self.key(n_id.item(), layer),
                        self.db_value_from_tensor(x),
                        disable_wal=True,
                    )
            else:
                for index, j in enumerate(n_id):
                    self.wbatch.put(
                        self.key(j.item(), layer), self.db_value_from_tensor(x[index])
                    )
                if self.multi_db:
                    self.db[layer].write(self.wbatch, disable_wal=True)
                    self.wbatch.clear()
                else:
                    self.db.write(self.wbatch, disable_wal=True)
                    self.wbatch.clear()
        else:
            # print("start push w/ offset,id")
            src_o = 0
            for (
                dst_o,
                c,
            ) in zip(offset.tolist(), count.tolist()):
                # print("filling writebatch")
                for i in range(c):
                    self.wbatch.put(
                        self.key(dst_o + i, layer),
                        self.db_value_from_tensor(x[src_o + i]),
                    )
                src_o += c
            # print("writing")
            if self.multi_db:
                self.db[layer].write(self.wbatch, disable_wal=True)
                self.wbatch.clear()
            else:
                self.db.write(self.wbatch, disable_wal=True)
                self.wbatch.clear()
            # print("writing done")

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError
