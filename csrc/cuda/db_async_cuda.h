#pragma once

#include <torch/extension.h>

void db_synchronize_cuda();
void db_read_async_cuda(int64_t layer,
                        torch::optional<torch::Tensor> optional_offset,
                        torch::optional<torch::Tensor> optional_count,
                        torch::Tensor index, torch::Tensor dst);
void db_write_async_cuda(int64_t layer, torch::Tensor src, torch::Tensor offset,
                      torch::Tensor count);
