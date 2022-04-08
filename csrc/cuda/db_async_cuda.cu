#include "db_async_cuda.h"
#include <iostream>
#include <ATen/cuda/CUDAContext.h>


#include "../thread.h"
#include "../gasdb.h"

extern GasDb * gasdb;


Thread &getThread() {
  static Thread thread;
  return thread;
}


void db_synchronize_cuda() {
  std::cout << "syncing thread\n";
  getThread().synchronize();
}

void db_read_async_cuda(int64_t layer, torch::optional<torch::Tensor> optional_offset,
                     torch::optional<torch::Tensor> optional_count,
                     torch::Tensor index, torch::Tensor dst) {


  std::cout << "db_read_async_cuda at layer " << layer << std::endl;
  AT_ASSERTM(!index.is_cuda(), "Index tensor must be a CPU tensor");
  AT_ASSERTM(dst.is_cuda(), "Target tensor must be a CUDA tensor");

  AT_ASSERTM(dst.is_contiguous(), "Target tensor must be contiguous");

  AT_ASSERTM(index.dim() == 1, "Index tensor must be one-dimensional");

  int64_t numel = 0;
  if (optional_offset.has_value()) {
    auto offset = optional_offset.value();
    AT_ASSERTM(!offset.is_cuda(), "Offset tensor must be a CPU tensor");
    AT_ASSERTM(offset.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(offset.dim() == 1, "Offset tensor must be one-dimensional");
    AT_ASSERTM(optional_count.has_value(), "Count tensor is undefined");
    auto count = optional_count.value();
    AT_ASSERTM(!count.is_cuda(), "Count tensor must be a CPU tensor");
    AT_ASSERTM(count.is_contiguous(), "Count tensor must be contiguous");
    AT_ASSERTM(count.dim() == 1, "Count tensor must be one-dimensional");
    AT_ASSERTM(offset.numel() == count.numel(), "Size mismatch");
    numel = count.sum().data_ptr<int64_t>()[0];
  }

  AT_ASSERTM(numel + index.numel() <= dst.size(0),
             "Target tensor size too small");

  auto stream = at::cuda::getCurrentCUDAStream(dst.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(dst.get_device()),
             "Asynchronous read requires a non-default CUDA stream");

  std::cout << "queuing new task in thread to populate src tensor then lauch cudaMemcpyAsync\n";
  AT_DISPATCH_ALL_TYPES(dst.scalar_type(), "db_read_async", [&] {
    getThread().run([=] {
      std::cout << "lauching populate src tensor then lauch cudaMemcpyAsync\n";
      auto options = torch::TensorOptions().dtype(torch::kFloat32);
      auto dst_data = dst.data_ptr<scalar_t>();

      int64_t numel_offset = 0;
      if (optional_offset.has_value())
        numel_offset = torch::sum(optional_count.value()).item<int64_t>();

      int64_t numel = numel_offset  + index.numel();
      int64_t esize = dst.numel()/dst.size(0);

      torch::Tensor src=torch::empty({numel, esize},options);
      auto src_data = src.data_ptr<scalar_t>();

      if (optional_offset.has_value()) {
        auto offset = optional_offset.value();
        auto count = optional_count.value();
        auto offset_data = offset.data_ptr<int64_t>();
        auto count_data = count.data_ptr<int64_t>();

        for (int64_t i = 0; i < offset.numel(); i++)
          for (int64_t j = 0; j < count_data[i]; j++)
            src.index_put_({offset_data[i]+j},gasdb->pull(offset_data[i]+j, layer));
      }
      for (int64_t i = 0; i < index.numel(); i++)
        src.index_put_({numel_offset+i},gasdb->pull(index[i].item<int64_t>(), layer));

      cudaMemcpyAsync(dst_data, src_data,
                      src.numel() * esize * sizeof(scalar_t),
                      cudaMemcpyHostToDevice, stream);

    });
  });
}

void db_write_async_cuda(int64_t layer, torch::Tensor src, torch::Tensor offset,
                      torch::Tensor count) {
  std::cout << "db_write_async_cuda at layer " << layer << std::endl;
  AT_ASSERTM(src.is_cuda(), "Source tensor must be a CUDA tensor");
  AT_ASSERTM(!offset.is_cuda(), "Offset tensor must be a CPU tensor");
  AT_ASSERTM(!count.is_cuda(), "Count tensor must be a CPU tensor");

  AT_ASSERTM(src.is_contiguous(), "Index tensor must be contiguous");
  AT_ASSERTM(offset.is_contiguous(), "Offset tensor must be contiguous");
  AT_ASSERTM(count.is_contiguous(), "Count tensor must be contiguous");

  AT_ASSERTM(offset.dim() == 1, "Offset tensor must be one-dimensional");
  AT_ASSERTM(count.dim() == 1, "Count tensor must be one-dimensional");
  AT_ASSERTM(offset.numel() == count.numel(), "Size mismatch");

  std::cout << "queuing new task in thread to  get cuda tensor then push to db\n";
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "db_write_async", [&] {
    getThread().run([=] {
      std::cout << "lauching  get cuda tensor then push to db\n";

      auto options = torch::TensorOptions().dtype(torch::kFloat32);
      int64_t numel = torch::sum(count).item<int64_t>();
      int64_t esize = src.numel()/src.size(0);
      torch::Tensor dst=torch::empty({numel, esize},options);

      auto dst_data = dst.data_ptr<scalar_t>();
      auto src_data = src.data_ptr<scalar_t>();

      cudaMemcpy(dst_data, src_data, numel*esize*sizeof(scalar_t), cudaMemcpyDeviceToHost);

      gasdb->push(dst, layer, offset, count);
    });
  });
}
