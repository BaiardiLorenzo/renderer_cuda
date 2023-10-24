#include "renderer.cuh"

Circle* generateCircles(std::size_t n, int width, int height, int minRadius, int maxRadius) {
    auto* circles = new Circle[n];
    std::mt19937 generator(777);

    std::uniform_int_distribution<int> colorDistribution(0, 255);
    std::uniform_int_distribution<int> pointXDistribution(1, width);
    std::uniform_int_distribution<int> pointYDistribution(1, height);
    std::uniform_int_distribution<int> radiusDistribution(minRadius, maxRadius);

#pragma omp parallel for default(none) shared(circles, generator)
    for (int i = 0; i < n; i++) {
        cv::Scalar color(colorDistribution(generator), colorDistribution(generator), colorDistribution(generator), 255);
        cv::Point center(pointXDistribution(generator), pointYDistribution(generator));
        int r = radiusDistribution(generator);
        circles[i] = Circle{color, center, r};
    }

    return circles;
}

double sequentialRenderer(Circle circles[], std::size_t nPlanes, std::size_t nCircles) {
    auto* planes = new cv::Mat[nPlanes];

    // START
    double start = omp_get_wtime();

    for (int i = 0; i < nPlanes; i++) {
        planes[i] = TRANSPARENT_MAT;
        for (int j = 0; j < nCircles; j++) {
            auto circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat result = sequentialCombinePlanes(planes, nPlanes);

    double time = omp_get_wtime() - start;
    // END

    delete[] planes;

    cv::imwrite(SEQ_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}

cv::Mat sequentialCombinePlanes(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int cn = result.channels();
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
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

double parallelRenderer(Circle circles[], std::size_t nPlanes, std::size_t nCircles) {
    auto* planes = new cv::Mat[nPlanes];

    // START
    double start = omp_get_wtime();

#pragma omp parallel for default(none) shared(planes, circles) firstprivate(nPlanes, nCircles)
    for (int i = 0; i < nPlanes; i++) {
        planes[i] = TRANSPARENT_MAT;
        for (int j = 0; j < nCircles; j++) {
            Circle circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat result = parallelCombinePlanes(planes, nPlanes);

    double time = omp_get_wtime() - start;
    // END

    delete[] planes;

    cv::imwrite(PAR_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}

cv::Mat parallelCombinePlanes(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int cn = result.channels();
#pragma omp parallel for default(none) shared(result, planes) firstprivate(nPlanes, cn) collapse(2)
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
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

double cudaRenderer(Circle circles[], std::size_t nPlanes, std::size_t nCircles) {
    auto* planes = new cv::Mat[nPlanes];

    // START
    double start = omp_get_wtime();

#pragma omp parallel for default(none) shared(planes, circles) firstprivate(nPlanes, nCircles)
    for (int i = 0; i < nPlanes; i++) {
        planes[i] = TRANSPARENT_MAT;
        for (int j = 0; j < nCircles; j++) {
            Circle circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat result = cudaCombinePlanes(planes, nPlanes);

    double time = omp_get_wtime() - start;
    // END

    delete[] planes;

    cv::imwrite(CUDA_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}

cv::Mat cudaCombinePlanes(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int width = result.cols;
    int height = result.rows;

    uchar4* d_resultData;
    uchar4* d_planesData;

    // INITIALIZATION OF GPU MEMORY
    cudaMalloc((void**)&d_resultData, width * height * sizeof(uchar4));
    cudaMalloc((void**)&d_planesData, width * height * sizeof(uchar4) * nPlanes);

    cudaMemcpy(d_resultData, result.data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    for (std::size_t i = 0; i < nPlanes; i++)
        cudaMemcpy(d_planesData + i * width * height, planes[i].data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

    // GRID AND BLOCK DIMENSIONS
    dim3 block(16, 16);
    dim3 grid((result.cols + block.x - 1) / block.x, (result.rows + block.y - 1) / block.y);

    // CUDA KERNEL
    cudaKernelCombinePlanes<<<grid, block>>>(d_resultData, d_planesData, result.cols, result.rows, (int) nPlanes);
    cudaDeviceSynchronize();

    // COPY RESULT FROM GPU TO CPU
    cudaMemcpy(result.data, d_resultData, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // FREE MEMORY
    cudaFree(d_planesData);
    cudaFree(d_resultData);

    return result;
}

__global__ void cudaKernelCombinePlanes(uchar4* resultData, const uchar4* planesData, int width, int height, int nPlanes) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        auto idx = y * width + x;
        for (int z = 0; z < nPlanes; z++) {
            auto idxP = z * width * height + idx;
            resultData[idx].x = resultData[idx].x * (1.0f - ALPHA) + planesData[idxP].x * ALPHA;
            resultData[idx].y = resultData[idx].y * (1.0f - ALPHA) + planesData[idxP].y * ALPHA;
            resultData[idx].z = resultData[idx].z * (1.0f - ALPHA) + planesData[idxP].z * ALPHA;
            resultData[idx].w = resultData[idx].w * (1.0f - ALPHA) + planesData[idxP].w * ALPHA;
        }
    }
}

