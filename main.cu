#include <vector>
#include "renderer.cuh"

#define MAX_TEST 100
#define MIN_TEST 10
#define N_CIRCLES 100

int main() {
    std::vector<double> sequentialTimes;
    std::vector<double> parallelTimes;
    std::vector<int> testPlanes;
    for (int i = MIN_TEST; i <= MAX_TEST; i += 10)
        testPlanes.push_back(i);

    for (int test: testPlanes) {
        const unsigned long long n = test * N_CIRCLES;
        Circle* circles = new Circle[n];
        generateCircles(circles, n);
        double t = rendererSequential(circles, test, N_CIRCLES);
        printf("SPEED %d: %f \n", test, t);
        parallelTimes.push_back(t);
        printf("\n");
        delete[] circles;
    }

    return 0;
}


