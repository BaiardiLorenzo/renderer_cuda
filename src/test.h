#ifndef RENDERER_CUDA_TEST_H
#define RENDERER_CUDA_TEST_H

// PATH RESULT FILE
#define RESULT_PATH ("../results/csv/result-"+std::to_string(WIDTH)+"-"+std::to_string(HEIGHT)+".csv")
#define SEQ_IMG_PATH "../results/img/seq/"
#define PAR_IMG_PATH "../results/img/par/"
#define CUDA_IMG_PATH "../results/img/cuda/"
#define CUDA_COLOR_IMG_PATH "../results/img/cuda_color/"

// IMAGES
#define WIDTH 1024
#define HEIGHT 1024
#define TRANSPARENT cv::Scalar(255, 255, 255, 0)
#define TRANSPARENT_MAT cv::Mat(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT)
#define ALPHA 0.5

// RADIUS CIRCLES
#define MAX_RADIUS 200
#define MIN_RADIUS 10

// FOR TESTING
#define MAX_TESTS 2000
#define MIN_TEST 200
#define SPACE 200
#define N_CIRCLES 100

#endif //RENDERER_CUDA_TEST_H
