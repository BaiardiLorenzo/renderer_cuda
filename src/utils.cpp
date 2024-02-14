//
// Created by thoma on 14/02/2024.
//

#include <fstream>
#include "utils.h"

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

void headerResultsCircle(const std::string& filename){
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open())
        outfile << "TEST_P;TEST_C;T_SEQ;T_PAR;SPEEDUP\n";
    outfile.close();
}

void exportResultsCircle(const std::string& filename, std::size_t testP, std::size_t testC, double tSeq, double tPar, double speedUp){
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::app);
    if(outfile.is_open()){
        outfile << std::fixed << std::setprecision(3);
        outfile << testP << ";" << testC << ";" << tSeq << ";" << tPar << ";" << speedUp << "\n";
    }
    outfile.close();
}

void headerResultsBlocks(const std::string& filename){
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open())
        outfile << "TEST;T_BLOCK_8;T_BLOCK_16;T_BLOCK_32\n";
    outfile.close();
}

void exportResultsBlocks(const std::string& filename, std::size_t test, double tSeq, double tBlock8, double tBlock16, double tBlock32){
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::app);
    if(outfile.is_open()){
        outfile << std::fixed << std::setprecision(3);
        outfile << test << ";" << tBlock8 << ";" << tBlock16 << ";" << tBlock32 << "\n";
    }
    outfile.close();
}

void headerResultsMemcpy(const std::string& filename){
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open())
        outfile << "TEST;T_SEQ;T_CUDA;SPEEDUP_CUDA;T_CUDA_MEMCPY;SPEEDUP_CUDA_MEMCPY\n";
    outfile.close();
}

void exportResultsMemcpy(const std::string& filename, std::size_t test, double tSeq, double tCuda, double speedUpCuda, double tCudaMemcpy, double speedUpCudaMemcpy){
    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::app);
    if(outfile.is_open()){
        outfile << std::fixed << std::setprecision(3);
        outfile << test << ";" << tSeq << ";" << tCuda << ";" << speedUpCuda << ";" << tCudaMemcpy << ";" << speedUpCudaMemcpy << "\n";
    }
    outfile.close();
}