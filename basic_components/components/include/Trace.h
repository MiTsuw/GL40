#ifndef TRACE_H
#define TRACE_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include "ConfigParams.h"
#include "Objectives.h"

#ifdef CUDA_CODE
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

using namespace std;

namespace components
{

class Trace
{
    //! Objectives to trace
    AMObjectives objs;

    //! Fichier de sortie avec valeurs de criteres et objectifs de la solution
    char* fileStats;
    //! Flux de sortie ouvert pour statistiques
    std::ofstream* OutputStream;

    //! Calcul duree d'execution
    time_t t0;
    //! Calcul duree d'execution
    time_t tf;
    //! Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
    double x0;
    //! Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
    double xf;

    // cuda timer
//    cudaEvent_t start, stop;
public:
    explicit Trace() : OutputStream(new ofstream) {}
    explicit Trace(AMObjectives objs) : objs(objs), OutputStream(new ofstream) {}

    //! \brief Destructeur.
    ~Trace(){
        delete OutputStream;
    }

    //! Initialisations
    void initialize(char* stats) {
        fileStats = stats;
        OutputStream->open(fileStats, ios::app);
        initialize();
    }

    void setObjs(AMObjectives objs) {
        this->objs = objs;
    }

    void initialize() {
        if (!OutputStream->rdbuf()->is_open())
        {
            cerr << "Unable to open file " << fileStats << "CRITICAL ERROR" << endl;
            exit(-1);
        }
        initHeaderStatistics(*OutputStream);

        time(&t0);
        x0 = clock();

#ifdef CUDA_CODE
        int devID = 0;
        cudaError_t error;
        cudaDeviceProp deviceProp;
        error = cudaGetDevice(&devID);
        if (error != cudaSuccess)
        {
            printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        }
        error = cudaGetDeviceProperties(&deviceProp, devID);
        if (deviceProp.computeMode == cudaComputeModeProhibited)
        {
            fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
            exit(EXIT_SUCCESS);
        }
        if (error != cudaSuccess)
        {
            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        }
        else
        {
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                   devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        }
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
#endif
    }

    void initHeaderStatistics(std::ostream& o) {
        o  	<< "iteration" << "\t"
            << "objective_1" << "\t"
            << "objective_2" << "\t"
            << "objective_3" << "\t"
            << "duree(s)" << "\t"
            << "duree(s.xx)" << "\t"
            << "cuda_duree(ms)" << endl;

    }//initHeaderStatistics

    void writeStatistics(int iteration, std::ostream& o) {

#ifdef CUDA_CODE
        // cuda timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
#endif
        o  	<< iteration << "\t"
            << objs[distr] << "\t"
            << objs[length] << "\t"
            << objs[sqr_length] << "\t"
            << time(&tf) - t0 << "\t"
            << (clock() - x0)/CLOCKS_PER_SEC << "\t"
#ifdef CUDA_CODE
            << elapsedTime << "\t"
#endif
            << endl;
    }//writeStatistics

    void writeStatistics(int iteration) {

        writeStatistics(iteration, *OutputStream);
    }

    void closeStatistics() {
        OutputStream->close();
    }

};
}//namespace components

#endif // TRACE_H
