#pragma once

#include <rocksdb/db.h>
#include <torch/script.h>

class GasDb {
public:
  GasDb(std::string path);
  ~GasDb();
  torch::Tensor pull(int64_t n_id, int layer, int edim);
  void push(torch::Tensor x,  int layer, torch::Tensor offset, torch::Tensor count);
  void reset();
private:
  rocksdb::DB* db;
  rocksdb::Options options;
  std::string key(int node, int layer);
  std::string db_value_from_tensor(torch::Tensor t);
  torch::Tensor tensor_from_db_value(std::string dbval);

};
