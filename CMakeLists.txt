cmake_minimum_required(VERSION 3.25)
project(renderer_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(renderer_cuda main.cu)

set_target_properties(renderer_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
