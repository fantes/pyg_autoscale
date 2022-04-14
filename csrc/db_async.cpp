#include <Python.h>
#include <torch/script.h>

#include "gasdb.h"
#include <iostream>
#include <sstream>

#ifdef WITH_CUDA
#include "cuda/db_async_cuda.h"
#endif

#include "./proto/tensor.pb.h"
#include "./proto/tensor.pb.cc"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__async(void) { return NULL; }
#endif

void db_synchronize_pull() {
#ifdef WITH_CUDA
  db_synchronize_pull_cuda();
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

void db_synchronize_push() {
#ifdef WITH_CUDA
  db_synchronize_push_cuda();
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

GasDb::GasDb(std::string path, int num_layers)
{
  auto cache = rocksdb::NewLRUCache(512 << 20);
  this->options.create_if_missing = true;
  this->options.error_if_exists = true;
  this->options.env->SetBackgroundThreads(4);
  this->options.compaction_style = rocksdb::kCompactionStyleLevel;
  this->options.write_buffer_size = 67108864; // 64MB
  this->options.max_write_buffer_number = 3;
  this->options.target_file_size_base = 67108864; // 64MB
  this->options.max_background_compactions = 4;
  this->options.level0_file_num_compaction_trigger = 8;
  this->options.level0_slowdown_writes_trigger = 17;
  this->options.level0_stop_writes_trigger = 24;
  this->options.num_levels = 4;
  this->options.max_bytes_for_level_base = 536870912; // 512MB
  this->options.max_bytes_for_level_multiplier = 8;

  this->options.compression = rocksdb::kLZ4Compression;
  this->options.bottommost_compression = rocksdb::kLZ4Compression;
  rocksdb::BlockBasedTableOptions table_options;
  table_options.block_cache = cache;
  table_options.block_size = 1 * 1024;
  table_options.cache_index_and_filter_blocks = true;
  table_options.pin_l0_filter_and_index_blocks_in_cache = true;
  table_options.format_version = 4;
  auto table_factory = rocksdb::NewBlockBasedTableFactory(table_options);
  this->options.table_factory.reset(table_factory);
  this->options.level_compaction_dynamic_level_bytes = true;
  this->options.max_background_compactions = 4;
  this->options.max_background_flushes = 2;
  this->options.bytes_per_sync = 1048576;
  this->options.compaction_pri = rocksdb::kMinOverlappingRatio;
  rocksdb::Status status = rocksdb::DB::Open(this->options, path, &this->db);
  assert(status.ok());
}

GasDb::~GasDb()
{
  rocksdb::Status status = this->db->Close();
  delete this->db;
  this->db = nullptr;
}

std::string GasDb::key(int node, int layer)
{
  //std::stringstream ss;
  //ss << layer << ":" << node;
  //return ss.str();
  return std::to_string(node);
}

std::string GasDb::db_value_from_tensor(torch::Tensor t)
{
  Vector v;
  *v.mutable_val() = {t.data_ptr<float>(), t.data_ptr<float>()+t.numel()};
  std::string st;
  v.SerializeToString(&st);
  return st;
  // std::stringstream st;
  // torch::save(t,st);
  // return st.str();
}

torch::Tensor GasDb::tensor_from_db_value(std::string dbval)
{
  Vector v;
  v.ParseFromString(dbval);
  std::vector<float> values(v.val().begin(), v.val().end());
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  return  torch::from_blob(values.data(), {(long)values.size()}, opts).clone();
  // std::stringstream st(dbval);
  // torch::Tensor t;
  // torch::load(t,st);
}


torch::Tensor GasDb::pull(int64_t n_id, int layer, int edim)
{
  std::string v;
  rocksdb::Status s = this->db->Get(rocksdb::ReadOptions(), this->key(n_id, layer), &v);
  if (s.IsNotFound())
          return torch::zeros({edim});
  torch::Tensor t = this->tensor_from_db_value(v);
  return t;
}


//void GasDb::push(torch::Tensor x, int layer, torch::Tensor ioffset, torch::Tensor icount)
void GasDb::push(torch::Tensor x, int layer, torch::Tensor ioffset, torch::Tensor icount)
{

  torch::Tensor offset = ioffset.to(torch::kInt64).contiguous();
  torch::Tensor count = icount.to(torch::kInt64).contiguous();
  int src_o = 0;
  rocksdb::WriteBatch batch;
  for (int i =0; i< offset.sizes()[0]; ++i)
    {
      int64_t c = count[i].item<int64_t>();
      for (int j=0; j<c; ++j)
        {
          batch.Put(this->key(offset[i].item<int64_t>()+j, layer), this->db_value_from_tensor(x[src_o+j]));
        }
      src_o += c;
    }
  rocksdb::WriteOptions wo;
  wo.disableWAL = true;
  wo.sync=false;
  rocksdb::Status s = this->db->Write(wo, &batch);
}


void db_read_async(int64_t layer,
                   torch::optional<torch::Tensor> optional_offset,
                   torch::optional<torch::Tensor> optional_count,
                   torch::Tensor index, torch::Tensor dst) {
#ifdef WITH_CUDA
  db_read_async_cuda(layer, optional_offset, optional_count, index, dst);
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

void db_write_async(int64_t layer, torch::Tensor src, torch::Tensor offset, torch::Tensor count) {
#ifdef WITH_CUDA
  db_write_async_cuda(layer, src, offset, count);
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

GasDb* gasdb = nullptr;
int num_push_threads = 25;

void init_db(std::string path, int64_t num_layers)
{
  num_push_threads = num_layers;
  gasdb = new GasDb(path,num_layers);
}

void delete_db()
{
  if (gasdb != nullptr)
    {
      delete(gasdb);
      gasdb = nullptr;
    }
}



void simple_push(torch::Tensor t, int64_t n_id, int64_t layer)
{
  gasdb->push(t,  layer, torch::tensor({n_id}), torch::tensor({1}));
}

torch::Tensor simple_pull( int64_t n_id, int64_t layer, int64_t edim)
{
  torch::Tensor t = gasdb->pull(n_id, layer,edim);
  return t;
}

static auto registry =
    torch::RegisterOperators()
  .op("torch_geometric_autoscale::db_synchronize_push", &db_synchronize_push)
  .op("torch_geometric_autoscale::db_synchronize_pull", &db_synchronize_pull)
  .op("torch_geometric_autoscale::db_read_async", &db_read_async)
  .op("torch_geometric_autoscale::db_write_async", &db_write_async)
  .op("torch_geometric_autoscale::init_db",&init_db)
  .op("torch_geometric_autoscale::delete_db",&delete_db)
  .op("torch_geometric_autoscale::simple_push", &simple_push)
  .op("torch_geometric_autoscale::simple_pull", &simple_pull)
  ;
