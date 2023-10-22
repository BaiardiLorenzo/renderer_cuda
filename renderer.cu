#include "renderer.cuh"

Circle* generateCircles(std::size_t n) {
    auto* circles = new Circle[n];
    std::mt19937 generator(777);

    std::uniform_int_distribution<int> colorDistribution(0, 255);
    std::uniform_int_distribution<int> pointXDistribution(1, WIDTH);
    std::uniform_int_distribution<int> pointYDistribution(1, HEIGHT);
    std::uniform_int_distribution<int> radiusDistribution(MIN_RADIUS, MAX_RADIUS);

#pragma omp parallel for default(none) shared(circles, generator)
    for (int i = 0; i < n; i++) {
        cv::Scalar color(colorDistribution(generator), colorDistribution(generator), colorDistribution(generator), 255);
        cv::Point center(pointXDistribution(generator), pointYDistribution(generator));
        int r = radiusDistribution(generator);
        circles[i] = Circle{color, center, r};
    }

    return circles;
}

double rendererSequential(Circle circles[], std::size_t nPlanes, std::size_t nCircles) {
    auto* planes = new cv::Mat[nPlanes];

    // START
    double start = omp_get_wtime();

    for (int i = 0; i < nPlanes; i++) {
        planes[i] = cv::Mat(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
        for (int j = 0; j < nCircles; j++) {
            auto circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat result = combinePlanesSequential(planes, nPlanes);

    double time = omp_get_wtime() - start;
    // END

    printf("Sequential time %f sec.\n", time);

    delete[] planes;

    cv::imwrite("../img/seq_" + std::to_string(nPlanes) + ".png", result);
    // cv::waitKey(0);
    return time;
}

cv::Mat combinePlanesSequential(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
    int cn = result.channels();
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            for (int z = 0; z < nPlanes; z++) {
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

double rendererParallel(Circle circles[], std::size_t nPlanes, std::size_t nCircles) {
    auto* planes = new cv::Mat[nPlanes];

    // START
    double start = omp_get_wtime();

#pragma omp parallel for default(none) shared(planes, circles) firstprivate(nPlanes, nCircles)
    for (int i = 0; i < nPlanes; i++) {
        planes[i] = cv::Mat(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
        for (int j = 0; j < nCircles; j++) {
            Circle circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat result = combinePlanesParallel(planes, nPlanes);

    double time = omp_get_wtime() - start;
    // END
    printf("Parallel time %f sec.\n", time);

    delete[] planes;

    cv::imwrite("../img/par_" + std::to_string(nPlanes) + ".png", result);
    return time;
}

cv::Mat combinePlanesParallel(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
    int cn = result.channels();
#pragma omp parallel for default(none) shared(result, planes) firstprivate(nPlanes, cn) collapse(2)
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            for (int z = 0; z < nPlanes; z++) {
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

double rendererCuda(Circle circles[], std::size_t nPlanes, std::size_t nCircles) {
    auto* planes = new cv::Mat[nPlanes];

    // START
    double start = omp_get_wtime();

#pragma omp parallel for default(none) shared(planes, circles) firstprivate(nPlanes, nCircles)
    for (int i = 0; i < nPlanes; i++) {
        planes[i] = cv::Mat(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);
        for (int j = 0; j < nCircles; j++) {
            Circle circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat result = combinePlanesCuda(planes, nPlanes);

    double time = omp_get_wtime() - start;
    // END
    printf("Cuda time %f sec.\n", time);

    delete[] planes;

    // cv::imshow("TEST", result);
    // cv::waitKey(0);

    cv::imwrite("../img/cuda_" + std::to_string(nPlanes) + ".png", result);
    return time;
}

cv::Mat combinePlanesCuda(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result(HEIGHT, WIDTH, CV_8UC4, TRANSPARENT);

    uchar4* d_resultData;
    uchar4* d_planesData;

    // Initialize pointers on GPU
    cudaMalloc((void**)&d_resultData, WIDTH * HEIGHT * sizeof(uchar4));
    cudaMalloc((void**)&d_planesData, WIDTH * HEIGHT * sizeof(uchar4) * nPlanes);

    cudaMemcpy(d_resultData, result.data, WIDTH * HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);
    for (std::size_t i = 0; i < nPlanes; i++) {
        uchar4* d_plane = d_planesData + i * WIDTH * HEIGHT;
        cudaMemcpy(d_plane, planes[i].data, WIDTH * HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);
    }

    // GRID AND BLOCK DIMENSIONS
    dim3 block(16, 16);
    dim3 grid((result.cols + block.x - 1) / block.x, (result.rows + block.y - 1) / block.y);

    // CUDA KERNEL
    combinePlanesKernel<<<grid, block>>>(d_resultData, d_planesData, result.cols, result.rows, (int)nPlanes);
    cudaDeviceSynchronize();

    // COPY RESULT FROM GPU TO CPU
    cudaMemcpy(result.data, d_resultData, WIDTH * HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // FREE MEMORY
    cudaFree(d_planesData);
    cudaFree(d_resultData);

    return result;
}

__global__ void combinePlanesKernel(uchar4* resultData, const uchar4* planesData, int width, int height, int nPlanes) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        auto idx = y * width + x;
        float4 result = make_float4(resultData[idx].x, resultData[idx].y, resultData[idx].z, resultData[idx].w);

        for (int z = 0; z < nPlanes; z++) {
            uchar4 plane = planesData[z * width * height + idx];
            result.x = result.x * (1.0f - ALPHA) + static_cast<float>(plane.x) * ALPHA;
            result.y = result.y * (1.0f - ALPHA) + static_cast<float>(plane.y) * ALPHA;
            result.z = result.z * (1.0f - ALPHA) + static_cast<float>(plane.z) * ALPHA;
            result.w = result.w * (1.0f - ALPHA) + static_cast<float>(plane.w) * ALPHA;
        }

        resultData[idx] = make_uchar4(static_cast<uchar>(result.x), static_cast<uchar>(result.y), static_cast<uchar>(result.z), static_cast<uchar>(result.w));
    }
}

