# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.27

# compile CUDA with C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe
# compile CXX with C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe
CUDA_DEFINES = 

CUDA_INCLUDES = --options-file CMakeFiles/renderer_cuda.dir/includes_CUDA.rsp

CUDA_FLAGS = -D_WINDOWS -Xcompiler=" /EHsc" -Xcompiler="-O2 -Ob2" -DNDEBUG -std=c++17 "--generate-code=arch=compute_52,code=[compute_52,sm_52]" -Xcompiler=-MD -openmp

CXX_DEFINES = 

CXX_INCLUDES = -external:IC:\src\opencv\build\install\include -external:W0

CXX_FLAGS = /DWIN32 /D_WINDOWS /EHsc -fopenmp -march=native /O2 /Ob2 /DNDEBUG -MD -openmp -openmp

