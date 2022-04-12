#include <Python.h>
#include <torch/script.h>

#include "gasdb.h"
#include <iostream>
#include <sstream>

#ifdef WITH_CUDA
#include "cuda/db_async_cuda.h"
#endif


#ifdef _WIN32
PyMODINIT_FUNC PyInit__async(void) { return NULL; }
#endif

void db_synchronize() {
#ifdef WITH_CUDA
  db_synchronize_cuda();
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

GasDb::GasDb(std::string path)
{
  //std::cout << "constructing gasdb\n" << std::flush;
  this->options.create_if_missing = true;
  this->options.error_if_exists = true;
  rocksdb::Status status = rocksdb::DB::Open(this->options, path, &this->db);
  assert(status.ok());
}

GasDb::~GasDb()
{
  //std::cout << "destructing gasdb\n" << std::flush;
  rocksdb::Status status = this->db->Close();
  delete this->db;
}

std::string GasDb::key(int node, int layer)
{
  std::stringstream ss;
  ss << layer << ":" << node;
  return ss.str();
}

std::string GasDb::db_value_from_tensor(torch::Tensor t)
{
  std::stringstream st;
  torch::save(t,st);
  return st.str();
}

torch::Tensor GasDb::tensor_from_db_value(std::string dbval)
{
  std::stringstream st(dbval);
  torch::Tensor t;
  torch::load(t,st);
  return t;
}


torch::Tensor GasDb::pull(int64_t n_id, int layer, int edim)
{
  std::string v;
  rocksdb::Status s = this->db->Get(rocksdb::ReadOptions(), this->key(n_id, layer), &v);
  if (s.IsNotFound())
          return torch::zeros({edim});
  torch::Tensor t =  this->tensor_from_db_value(v);
  return t;
}


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
  rocksdb::Status s = this->db->Write(rocksdb::WriteOptions(), &batch);
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

GasDb * gasdb = nullptr;
std::string gasdbpath="/tmp/dbhist.db";

void init_db(std::string path)
{
  if (!gasdb)
    delete(gasdb);
  gasdbpath = path;
  gasdb = new GasDb(gasdbpath);
}

void delete_db()
{
  delete(gasdb);
}



void simple_push(torch::Tensor t, int64_t n_id, int64_t layer)
{
  gasdb->push(t,  layer, torch::tensor({n_id}), torch::tensor({1}));
}

void simple_pull( int64_t n_id, int64_t layer, torch::Tensor t, int64_t edim)
{
  t.copy_(gasdb->pull(n_id, layer,edim));
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_geometric_autoscale::db_synchronize", &db_synchronize)
        .op("torch_geometric_autoscale::db_read_async", &db_read_async)
        .op("torch_geometric_autoscale::db_write_async", &db_write_async)
  .op("torch_geometric_autoscale::init_db",&init_db)
  .op("torch_geometric_autoscale::delete_db",&delete_db)
  .op("torch_geometric_autoscale::simple_push", &simple_push)
  .op("torch_geometric_autoscale::simple_pull", &simple_pull)
  ;
