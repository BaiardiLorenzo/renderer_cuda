#include "src/renderer.cuh"
#include "src/test.h"
#include <map>
#include <iomanip>

void headerResults(const std::string& filename, int nThreads){
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open())
        outfile << "TEST;T_SEQ;";
    for(int i=2; i<=nThreads; i+=2)
        outfile << "T_PAR" << i << ";SPEEDUP" << i << ";";
    outfile << "T_CUDA;SPEEDUP_CUDA;T_CUDA_COLOR;SPEEDUP_CUDA_COLOR\n";
    outfile.close();
}

void exportResults(const std::string& filename, std::size_t test, double tSeq, const std::map<std::size_t, double>& tPars,
                   std::map<std::size_t,double> speedUps, double tCuda, double speedUpCuda, double tCudaColor=-1, double speedUpCudaColor=-1){
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::app);
    if(outfile.is_open()){
        outfile << std::fixed << std::setprecision(3);
        outfile << test << ";" << tSeq << ";";
        for(auto tPar: tPars)
            outfile << tPar.second << ";" << speedUps[tPar.first] << ";";
        outfile << tCuda << ";" << speedUpCuda << ";" << tCudaColor << ";" << speedUpCudaColor << "\n";
    }
    outfile.close();
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
        auto circles = generateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        auto planes = generatePlanes(test, circles, N_CIRCLES);

        printf("\nTEST PLANES: %llu\n", test);

        // TEST SEQUENTIAL
        double tSeq = sequentialRenderer(planes, test);
        printf("SEQUENTIAL Time: %f\n", tSeq);
/*
        // TEST PARALLEL
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
        */

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
        //exportResults(RESULT_PATH, test, tSeq, tPars, speedUps, tCuda, speedUpCuda, tCudaColor, speedUpCudaColor);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;
    }
    return 0;
}
