//
// Created by thoma on 14/02/2024.
//

#include <map>
#include <iomanip>

#ifndef RENDERER_CUDA_UTILS_H
#define RENDERER_CUDA_UTILS_H

void headerResults(const std::string& filename, int nThreads);

void exportResults(const std::string& filename, std::size_t test, double tSeq, const std::map<std::size_t, double>& tPars,
                   std::map<std::size_t,double> speedUps, double tCuda, double speedUpCuda, double tCudaColor, double speedUpCudaColor);

void headerResultsCircle(const std::string& filename);

void exportResultsCircle(const std::string& filename, std::size_t testP, std::size_t testC, double tSeq, double tPar, double speedUp);

void headerResultsBlocks(const std::string& filename);

void exportResultsBlocks(const std::string& filename, std::size_t test, double tSeq, double tBlock8, double tBlock16, double tBlock32);

void headerResultsMemcpy(const std::string& filename);

void exportResultsMemcpy(const std::string& filename, std::size_t test, double tSeq, double tCuda, double speedUpCuda, double tCudaMemcpy, double speedUpCudaMemcpy);

#endif //RENDERER_CUDA_UTILS_H
