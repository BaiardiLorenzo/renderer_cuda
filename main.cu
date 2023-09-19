#include <vector>
#include "renderer.cuh"

#define MAX_TESTS 100
#define MIN_TEST 10
#define N_CIRCLES 100

int main() {
    std::vector<double> sequentialTimes;
    std::vector<double> parallelTimes;
    std::vector<int> testPlanes;
    for (int i = MIN_TEST; i <= MAX_TESTS; i += 10)
        testPlanes.push_back(i);

    for (int test: testPlanes) {
        Circle circles[1000];
        generateCircles(circles, 1000);
        double t = rendererParallel(circles, test, N_CIRCLES);
        printf("SPEED %d: %f \n", test, t);
        parallelTimes.push_back(t);
        printf("\n");
    }

    return 0;
}


