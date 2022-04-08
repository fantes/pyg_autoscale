0 setup cuda
0.5 env vars
export CUDA_HOME=/usr/local/cuda-11.4
export CUDA_PATH=/usr/local/cuda-11.4
export CPATH=/usr/local/cuda-11.4/include
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=~/.local/lib/:~/.local/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
1 install pytorch from source
GLIBCXX_USE_CXX11_ABI=1 python3 ./setup.py install --user
1.5 install torchvision from source
2 install metis from source
(INTIDX to 64)
3 install torch_sparse from source
4 install torch_scatter from source
6 install torch_cluster from source
7 install torch_spline_conv from source
8 install pyg_autoscale
