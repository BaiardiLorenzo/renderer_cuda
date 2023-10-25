#include "src/renderer.cuh"
#include "src/test.h"
#include <map>

void headerResults(const std::string& filename, int nThreads){
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open())
        outfile << "test;tSeq;";
    for(int i=2; i<=nThreads; i+=2)
        outfile << "tPar" << i << ";speedUp" << i << ";";
    outfile << "tCuda;speedUpCuda\n";
    outfile.close();
}

void exportResults(const std::string& filename, std::size_t test, double tSeq, const std::map<std::size_t, double>& tPars,
                   std::map<std::size_t,double> speedUps, double tCuda, double speedUpCuda){
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::app);
    if(outfile.is_open()){
        outfile << test << ";" << tSeq << ";";
        for(auto tPar: tPars)
            outfile << tPar.second << ";" << speedUps[tPar.first] << ";";
        outfile << tCuda << ";" << speedUpCuda << "\n";
    }
    outfile.close();
}


int main() {
#ifdef _OPENMP
    printf("**Number of cores/threads: %d**\n", omp_get_num_procs());
    omp_set_dynamic(0);
#endif
    headerResults(TEST_PATH, omp_get_num_procs());
    std::vector<std::size_t> testPlanes;
    for (std::size_t i = MIN_TEST; i <= MAX_TESTS; i += SPACE)
        testPlanes.push_back(i);

    for (auto test: testPlanes) {
        // GENERATION OF CIRCLES
        auto circles = generateCircles(test * N_CIRCLES, WIDTH, HEIGHT, MIN_RADIUS, MAX_RADIUS);
        auto planes = generatePlanes(test, circles, N_CIRCLES);

        printf("\nTEST: %llu\n", test);

        // TEST SEQUENTIAL
        double tSeq = sequentialRenderer(planes, test);
        printf("Sequential time: %f\n", tSeq);

        // TEST PARALLEL
        std::map<std::size_t, double> tPars;
        std::map<std::size_t, double> speedUps;
        for (int i=2; i<=omp_get_num_procs(); i+=2) {
            //SET NUMBER OF THREADS
            omp_set_num_threads(i);

            // TEST PARALLEL
            double tPar = parallelRenderer(planes, test);
            printf("Parallel time with %d threads: %f\n", i, tPar);

            double speedUp = tSeq / tPar;
            printf("Speedup with %d threads: %f \n", i, speedUp);

            // SAVE RESULTS
            tPars.insert(std::pair<std::size_t, double>(i, tPar));
            speedUps.insert(std::pair<std::size_t, double>(i, speedUp));
        }

        // TEST CUDA
        double tCuda = cudaRenderer(planes, test);
        printf("Cuda time: %f\n", tCuda);

        double speedUpCuda = tSeq / tCuda;
        printf("Speedup Cuda: %f \n\n", speedUpCuda);

        //WRITE RESULTS TO TXT FILE
        exportResults(TEST_PATH,test,tSeq,tPars,speedUps,tCuda,speedUpCuda);

        // DELETE ARRAY DYNAMIC ALLOCATED
        delete[] circles;
        delete[] planes;
    }
    return 0;
}
