#include "renderer.cuh"

Circle* generateCircles(std::size_t n, int width, int height, int minRadius, int maxRadius) {
    auto* circles = new Circle[n];
    std::mt19937 generator(std::random_device{}());

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

cv::Mat* generatePlanes(std::size_t nPlanes, Circle circles[], std::size_t nCircles) {
    auto *planes = new cv::Mat[nPlanes];

#pragma omp parallel for default(none) shared(planes, circles) firstprivate(nPlanes, nCircles)
    for (int i = 0; i < nPlanes; i++) {
        planes[i] = TRANSPARENT_MAT;
        for (int j = 0; j < nCircles; j++) {
            auto circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
        }
    }

    return planes;
}


double sequentialRenderer(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int cn = result.channels();

    // START
    double start = omp_get_wtime();

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

    double time = omp_get_wtime() - start;
    // END

    cv::imwrite(SEQ_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}


double parallelRenderer(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int cn = result.channels();

    // START
    double start = omp_get_wtime();

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

    double time = omp_get_wtime() - start;
    // END

    cv::imwrite(PAR_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}


double cudaRenderer(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int width = result.cols;
    int height = result.rows;

    uchar4* d_resultData;
    uchar4* d_planesData;

    // INITIALIZATION OF GPU MEMORY
    cudaMalloc((void**)&d_resultData, width * height * sizeof(uchar4));
    auto planesMalloc = cudaMalloc((void**)&d_planesData, width * height * sizeof(uchar4) * nPlanes);
    if (planesMalloc != cudaSuccess) {
        std::cout << "ERROR: allocating memory for planes: " << cudaGetErrorString(planesMalloc) << std::endl;
        // FREE MEMORY
        cudaFree(d_planesData);
        cudaFree(d_resultData);
        return -1;
    }

    // CUDA MEMORY COPY
    cudaMemcpy(d_resultData, result.data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    for (std::size_t i = 0; i < nPlanes; i++)
        cudaMemcpy(d_planesData + i * width * height, planes[i].data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

    // GRID AND BLOCK DIMENSIONS
    dim3 block(32, 32); // threads x block
    dim3 grid(result.cols / block.x, result.rows/ block.y); // blocks

    // START
    double start = omp_get_wtime();

    // CUDA KERNEL
    cudaKernelCombinePlanes<<<grid, block>>>(d_resultData, d_planesData,result.cols, result.rows, (int) nPlanes);
    cudaDeviceSynchronize();

    double time = omp_get_wtime() - start;
    // END

    // COPY RESULT FROM GPU TO CPU
    cudaMemcpy(result.data, d_resultData, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // FREE MEMORY
    cudaFree(d_planesData);
    cudaFree(d_resultData);

    cv::imwrite(CUDA_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}

__global__ void cudaKernelCombinePlanes(uchar4* resultData, const uchar4* planesData, int width, int height, int nPlanes) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("tx: %d, ty: %d, bx: %d, by: %d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    if (x < width && y < height) {
        auto idx = y * width + x;
        auto oneMinusAlpha = 1.0f - ALPHA;
        auto result = resultData[idx];

        for (int z = 0; z < nPlanes; z++) {
            auto idxP = z * width * height + idx;
            const auto &plane = planesData[idxP];

            result.x = result.x * oneMinusAlpha + plane.x * ALPHA;
            result.y = result.y * oneMinusAlpha + plane.y * ALPHA;
            result.z = result.z * oneMinusAlpha + plane.z * ALPHA;
            result.w = result.w * oneMinusAlpha + plane.w * ALPHA;
        }
        resultData[idx] = result;
        //printf("%d, %d, %d, %d\n", resultData[idx].x, resultData[idx].y, resultData[idx].z, resultData[idx].w);
    }
}

double cudaRendererColor(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int width = result.cols;
    int height = result.rows;
    int channels = 4;

    auto *planesData = new uchar[height * width * nPlanes * channels];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++){
            for (int c = 0; c < channels; c++){
                for (int z = 0; z < nPlanes; z++)
                    planesData[i * width * channels * nPlanes + j * channels * nPlanes + c * nPlanes + z] =
                            planes[z].data[i * planes[z].step + j * channels + c];
            }
        }
    }

    // INITIALIZATION OF GPU MEMORY
    uchar4* d_resultData;
    uchar* d_planesData;

    cudaMalloc((void**)&d_resultData, width * height * sizeof(uchar4));
    cudaMemcpy(d_resultData, result.data, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);

    auto planesMalloc = cudaMalloc((void**)&d_planesData, width * height * sizeof(uchar) * nPlanes * channels);
    if (planesMalloc != cudaSuccess) {
        std::cout << "Error allocating memory for planes: " << cudaGetErrorString(planesMalloc) << std::endl;
        // FREE MEMORY
        cudaFree(d_planesData);
        cudaFree(d_resultData);
        return -1;
    }
    cudaMemcpy(d_planesData, planesData, width * height * sizeof(uchar) * nPlanes * channels, cudaMemcpyHostToDevice);

    // GRID AND BLOCK DIMENSIONS
    dim3 block(32, 32); // each thread manages a color across all planes
    dim3 grid(result.rows / block.x, result.cols / block.y); // each block manages a pixel across all planes

    // START
    double start = omp_get_wtime();

    // CUDA KERNEL
    cudaKernelCombinePlanesColor<<<grid, block, nPlanes * channels * sizeof(uchar)>>>(d_resultData, d_planesData, result.cols, result.rows, (int) nPlanes);
    cudaDeviceSynchronize();

    double time = omp_get_wtime() - start;
    // END

    // COPY RESULT FROM GPU TO CPU
    cudaMemcpy(result.data, d_resultData, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // FREE MEMORY
    cudaFree(d_resultData);
    cudaFree(d_planesData);

    cv::imwrite(CUDA_COLOR_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    return time;
}

__global__ void cudaKernelCombinePlanesColor(uchar4* d_resultData, const uchar* d_planesData,
                                              const int width, const int height, const int nPlanes) {

    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        auto oneMinusAlpha = 1.0f - ALPHA;
        int channels = 4;
        auto idx = y * width * channels * nPlanes + x * channels * nPlanes;
        auto idxP = y * width + x;
        uchar threadData[] = {d_resultData[idxP].x, d_resultData[idxP].y, d_resultData[idxP].z, d_resultData[idxP].w};

        for (int c = 0; c < channels; c++){
            for (int z = 0; z < nPlanes; z++) {
                threadData[c] = threadData[c] * oneMinusAlpha + d_planesData[idx + c * nPlanes + z] * ALPHA;
            }
        }
        d_resultData[idxP] = {threadData[0], threadData[1], threadData[2], threadData[3]};
    }

}
