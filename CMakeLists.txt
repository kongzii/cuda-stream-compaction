cmake_minimum_required(VERSION 3.10)
project(CSC)

# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 -ccbin gcc-6)

find_package(CUDA REQUIRED)

cuda_add_executable(CSC src/main.cu src/data.hpp src/utils.hpp src/Config/Config.hpp src/Config/Config.cpp)

