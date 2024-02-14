#include "src/renderer.cuh"
#include "src/test.h"
#include "src/utils.h"
#include <map>
#include <iomanip>

void testCudaMemcpy(std::vector<std::size_t> testPlanes){
    printf("TEST CUDA MEMCPY");
    for (auto test: testPlanes) { // 1000, 10000 (256x256, 512x512, 1024x1024)
        printf("TEST PLANES: %llu\n", test);

        // GENERATION OF CIRCLES
        auto circles = parallelGenerateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        auto planes = parallelGeneratePlanes(test, circles, N_CIRCLES);

        // TEST SEQUENTIAL
        double tSeq = sequentialRenderer(planes, test);
        printf("SEQUENTIAL Time: %f\n", tSeq);

        // TEST CUDA BLOCKS
        double tCuda = cudaRenderer(planes, test, 8);
        printf("CUDA Time: %f\n", tCuda);

        double speedUpCuda = tSeq / tCuda;
        printf("CUDA-COLOR Speedup: %f\n\n", speedUpCuda);

        double tCudaMemcpy = cudaRenderer(planes, test, 16);
        printf("CUDA Time: %f\n", tCudaMemcpy);

        double speedUpCudaMemcpy = tSeq / tCudaMemcpy;
        printf("CUDA-COLOR Speedup: %f\n\n", speedUpCudaMemcpy);

        // WRITE RESULTS TO TXT FILE
        exportResultsMemcpy(RESULT_MEMCPY_PATH, test, tSeq, tCuda, tCudaMemcpy);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;
    }
}

void testCudaBlocks(testPlanes){
    printf("TEST CUDA GRID");
    for (auto test: testPlanes) { // 500, 5000 (256x256, 512x512, 1024x1024)
        printf("TEST PLANES: %llu\n", test);

        // GENERATION OF CIRCLES
        auto circles = parallelGenerateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        auto planes = parallelGeneratePlanes(test, circles, N_CIRCLES);

        // TEST SEQUENTIAL
        double tSeq = sequentialRenderer(planes, test);
        printf("SEQUENTIAL Time: %f\n", tSeq);

        // TEST CUDA BLOCKS
        double tBlock8 = cudaRenderer(planes, test, 8);
        printf("CUDA Time: %f\n", tBlock8);

        double tBlock16 = cudaRenderer(planes, test, 16);
        printf("CUDA Time: %f\n", tBlock16);

        double tBlock32 = cudaRenderer(planes, test, 32);
        printf("CUDA Time: %f\n", tBlock32);

        // WRITE RESULTS TO TXT FILE
        exportResultsBlocks(RESULT_BLOCKS_PATH, test, tSeq, tBlock8, tBlock16, tBlock32);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;
    }
}

void testCircles(std::vector<std::size_t> testPlanes){
    printf("TEST CIRCLES");       // n = 50, 500
    for (auto test: testPlanes) { // N = 100, 1000, 10000 (256x256, 512x512, 1024x1024)
        printf("TEST PLANES: %llu\n", test);
        // GENERATION OF CIRCLES
        double start = omp_get_wtime();
        auto circles = sequentialGenerateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        auto planes = sequentialGeneratePlanes(test, circles, N_CIRCLES);
        double seqTime = omp_get_wtime() - start;
        printf("SEQUENTIAL Time: %f\n", seqTime);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;

        // GENERATION OF CIRCLES
        start = omp_get_wtime();
        circles = parallelGenerateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        planes = parallelGeneratePlanes(test, circles, N_CIRCLES);
        double parTime = omp_get_wtime() - start;
        printf("PARALLEL Time: %f\n", parTime);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;

        // WRITE RESULTS TO CSV FILE
        exportResultsCircle(RESULT_CIRCLES_PATH, test, seqTime, parTime, seqTime/parTime);
    }
}

int main() {
#ifdef _OPENMP
    printf("**OPENMP :: Number of cores/threads: %d**\n", omp_get_num_procs());
    omp_set_dynamic(0);
#endif
    cudaDeviceProp device{};
    cudaGetDeviceProperties(&device, 0);
    printf("**CUDA :: MultiProcessorCount: %d**\n", device.multiProcessorCount); // 6
    printf("**CUDA :: Max Threads per MultiProcessor: %d**\n", device.maxThreadsPerMultiProcessor); // 2048
    printf("**CUDA :: Max Threads per block: %d**\n", device.maxThreadsPerBlock); // 1024
    printf("**CUDA :: Max Blocks per MultiProcessor: %d**\n", device.maxBlocksPerMultiProcessor); // 32
    // MAXTHREADS TOTAL: 12288


    headerResults(RESULT_PATH, omp_get_num_procs());
    std::vector<std::size_t> testPlanes;
    for (std::size_t i = MIN_TEST; i <= MAX_TESTS; i += SPACE)
        testPlanes.push_back(i);

    for (auto test: testPlanes) {
        // GENERATION OF CIRCLES
        auto circles = parallelGenerateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        auto planes = parallelGeneratePlanes(test, circles, N_CIRCLES);

        printf("\nTEST PLANES: %llu\n", test);

        // TEST SEQUENTIAL
        double tSeq = sequentialRenderer(planes, test);
        printf("SEQUENTIAL Time: %f\n", tSeq);

        // TEST OPENMP
        std::map<std::size_t, double> tPars;
        std::map<std::size_t, double> speedUps;
        for (int i=2; i<=omp_get_num_procs(); i+=2) {
            // SET NUMBER OF THREADS
            omp_set_num_threads(i);

            // TEST PARALLEL
            double tPar = parallelRenderer(planes, test);
            printf("PARALLEL-%d Time: %f\n", i, tPar);

            double speedUp = tSeq / tPar;
            printf("PARALLEL-%d Speedup: %f \n", i, speedUp);

            // SAVE RESULTS
            tPars.insert(std::pair<std::size_t, double>(i, tPar));
            speedUps.insert(std::pair<std::size_t, double>(i, speedUp));
        }

        // TEST CUDA
        double tCuda = cudaRenderer(planes, test);
        printf("CUDA Time: %f\n", tCuda);

        double speedUpCuda = tSeq / tCuda;
        printf("CUDA Speedup: %f\n", speedUpCuda);

        // TEST CUDA COLOR
        double tCudaColor = cudaRendererColor(planes, test);
        printf("CUDA-COLOR Time: %f\n", tCudaColor);

        double speedUpCudaColor = tSeq / tCudaColor;
        printf("CUDA-COLOR Speedup: %f\n\n", speedUpCudaColor);

        // WRITE RESULTS TO TXT FILE
        exportResults(RESULT_PATH, test, tSeq, tPars, speedUps, tCuda, speedUpCuda, tCudaColor, speedUpCudaColor);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;
    }

    testCudaBlocks(testPlanes);

    testCudaMemcpy(testPlanes);

    testCircles(testPlanes);

    return 0;
}
