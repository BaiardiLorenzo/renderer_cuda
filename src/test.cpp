//cv::Mat* generatePlanesWeighted(std::size_t nPlanes, Circle circles[], std::size_t nCircles);
//double sequentialRendererWeighted(cv::Mat planes[], std::size_t nPlanes);

/*
cv::Mat* generatePlanesWeighted(std::size_t nPlanes, Circle circles[], std::size_t nCircles) {
    auto *planes = new cv::Mat[nPlanes];

#pragma omp parallel for default(none) shared(planes, circles) firstprivate(nPlanes, nCircles)
    for (int i = 0; i < nPlanes; i++) {
        planes[i] = TRANSPARENT_MAT;
        auto background = TRANSPARENT_MAT;
        for (int j = 0; j < nCircles; j++) {
            planes[i].copyTo(background);
            auto circle = circles[i * nCircles + j];
            cv::circle(planes[i], circle.center, circle.r, circle.color, cv::FILLED, cv::LINE_AA);
            cv::addWeighted(planes[i], ALPHA, background, 1 - ALPHA, 0, planes[i]);
        }
    }
#pragma omp barrier

    return planes;
}
*/

/*
double sequentialRendererWeighted(cv::Mat planes[], std::size_t nPlanes) {
    cv::Mat result = TRANSPARENT_MAT;
    int cn = result.channels();

    // START
    double start = omp_get_wtime();

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            for (int z = 0; z < nPlanes; z++) {
                cv::Mat *src2 = &planes[z];
                double alphaSrc = (double)(src2->data[i * src2->step + j * cn + 3]) / 255;
                double alphaRes = (double)(result.data[i * result.step + j * cn + 3]) / 255;
                double alpha = alphaSrc + alphaRes * (1 - alphaSrc);
                for (int c = 0; c < cn - 1; c++)
                    result.data[i * result.step + cn * j + c] = (src2->data[i * src2->step + j * cn + c] * alphaSrc + result.data[i * result.step + j * cn + c] * alphaRes * (1-alphaSrc)) / alpha;
                result.data[i * result.step + cn * j + 3] = alpha * 255;
            }
        }
    }

    double time = omp_get_wtime() - start;
    // END

    cv::imwrite(SEQ_IMG_PATH + std::to_string(nPlanes) + ".png", result);
    //cv::imshow("seq"+std::to_string(nPlanes), result);
    //cv::waitKey(0);
    return time;
}
*/
