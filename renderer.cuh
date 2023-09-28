#ifndef RENDERER_CUDA_RENDERER_CUH
#define RENDERER_CUDA_RENDERER_CUH

#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#define WIDTH 1024
#define HEIGHT 1024
#define TRANSPARENT cv::Scalar(255, 255, 255, 0)
#define ALPHA 0.4

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

#endif //RENDERER_CUDA_RENDERER_CUH
