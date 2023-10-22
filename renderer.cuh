#ifndef RENDERER_CUDA_RENDERER_CUH
#define RENDERER_CUDA_RENDERER_CUH

#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#define WIDTH 1920
#define HEIGHT 1080
#define TRANSPARENT cv::Scalar(255, 255, 255, 0)
#define ALPHA 0.5f

#define MAX_RADIUS 200
#define MIN_RADIUS 100

struct Circle {
    cv::Scalar color;
    cv::Point center;
    int r;
};

Circle* generateCircles(std::size_t n);

double rendererSequential(Circle circles[], std::size_t nPlanes, std::size_t nCircles);

cv::Mat combinePlanesSequential(cv::Mat planes[], std::size_t nPlanes);

double rendererParallel(Circle circles[], std::size_t nPlanes, std::size_t nCircles);

cv::Mat combinePlanesParallel(cv::Mat planes[], std::size_t nPlanes);

double rendererCuda(Circle circles[], std::size_t nPlanes, std::size_t nCircles);

cv::Mat combinePlanesCuda(cv::Mat planes[], std::size_t nPlanes);

__global__ void combinePlanesKernel(uchar* resultData, uchar** planesData, int width, int height, int nPlanes, int cn);


#endif //RENDERER_CUDA_RENDERER_CUH
