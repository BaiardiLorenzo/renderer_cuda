//
// Created by loreb on 07/06/2023.
//

#ifndef RENDERER_CUDA_RENDERER_CUH
#define RENDERER_CUDA_RENDERER_CUH


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

void generateCircles(Circle circles[], unsigned long long n);

double rendererParallel(Circle circles[], unsigned long long nPlanes, unsigned long long nCircles);

cv::Mat combinePlanesParallel(cv::Mat planes[], unsigned long long nPlanes);


#endif //RENDERER_CUDA_RENDERER_CUH
