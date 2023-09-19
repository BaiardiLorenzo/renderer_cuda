//
// Created by loreb on 07/06/2023.
//


#include "renderer.cuh"

void generateCircles(Circle circles[], unsigned long long n) {
    std::srand(123);
    for (int i = 0; i < n; i++) {
        cv::Scalar color(std::rand() % 256, std::rand() % 256, std::rand() % 256, 255);
        cv::Point center(std::rand() % HEIGHT + 1, std::rand() % WIDTH + 1);
        int r = std::rand() % (MAX_RADIUS - MIN_RADIUS) + MIN_RADIUS + 1;
        circles[i] = {color, center, r};
    }
}

double rendererParallel(Circle circles[], unsigned long long nPlanes, unsigned long long nCircles) {
    printf("RENDERER PARALLEL %llu: ", nPlanes);
    cv::Mat planes[10];

    //double start = omp_get_wtime();

    for (int i = 0; i < 10; i++) {
        planes[i] = cv::Mat(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
        for (int j = 0; j < nCircles; j++) {
            Circle circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }
    printf("ok ");


    cv::Mat result = combinePlanesParallel(planes, 1000);

    //double time = omp_get_wtime() - start;

    printf(" TIME %d sec.\n", 0);
    cv::imshow("test", result);
    cv::waitKey();
    return 0;
}

cv::Mat combinePlanesParallel(cv::Mat planes[], unsigned long long nPlanes) {
    cv::Mat result(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
    int cn = result.channels();
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            for (int z = 0; z < 10; z++) {
                cv::Mat *src2 = &planes[z];
                for (int c = 0; c < cn; c++)
                    result.data[i * result.step + cn * j + c] =
                            result.data[i * result.step + j * cn + c] * (1 - ALPHA) +
                            src2->data[i * src2->step + j * cn + c] * (ALPHA);
            }
        }
    }
    return result;
}