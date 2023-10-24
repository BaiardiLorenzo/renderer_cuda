#ifndef RENDERER_CUDA_RENDERER_CUH
#define RENDERER_CUDA_RENDERER_CUH

#include "test.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct Circle {
    cv::Scalar color;
    cv::Point center;
    int r;
};

Circle* generateCircles(std::size_t n, int width, int height, int minRadius, int maxRadius);

double sequentialRenderer(Circle circles[], std::size_t nPlanes, std::size_t nCircles);

cv::Mat sequentialCombinePlanes(cv::Mat planes[], std::size_t nPlanes);

double parallelRenderer(Circle circles[], std::size_t nPlanes, std::size_t nCircles);

cv::Mat parallelCombinePlanes(cv::Mat planes[], std::size_t nPlanes);

double cudaRenderer(Circle circles[], std::size_t nPlanes, std::size_t nCircles);

cv::Mat cudaCombinePlanes(cv::Mat planes[], std::size_t nPlanes);

__global__ void cudaKernelCombinePlanes(uchar4* resultData, const uchar4* planesData, int width, int height, int nPlanes);


#endif //RENDERER_CUDA_RENDERER_CUH
