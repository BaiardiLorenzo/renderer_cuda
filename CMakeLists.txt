cmake_minimum_required(VERSION 3.24)
project(renderer_cuda CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(OpenCV_DIR "C:\\src\\opencv\\build\\install")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(renderer_cuda main.cu renderer.cu renderer.cuh)
target_link_libraries(renderer_cuda ${OpenCV_LIBS} "${OpenMP_CXX_FLAGS}")
target_compile_options(renderer_cuda PRIVATE "${OpenMP_CXX_FLAGS}")
set_target_properties(renderer_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

