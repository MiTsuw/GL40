#ifndef ADAPTATOR_BASICS_H
#define ADAPTATOR_BASICS_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

#include "random_generator.h"



//#include <helper_functions.h>


#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"

#include "SpiralSearch.h"
#include "NIter.h"
#include "ViewGrid.h"

#ifdef CUDA_CODE
#include <cuda_runtime.h>"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

#define SOM3D   1

#define LARG_CHAP 0.4f // chapeau mexicain for triggerring
#define DELTA 1.0f   // for controlling the number of activated cells in each iteration

//! HW 29/03/15 : add #define
#define MAX_RAND_BUFFER_SIZE 64
//! HW 30/03/15 : add #define
//! In order to get a unique cell id (coordinate in the cellular matrix) according
//! to the cell::PC, here a hypothetical cellular matrix width is defined.
//! Note that this hypothetically computed cell id is only used for GPU CUDA random
//! number seed/starting state setup.
#define MAX_CM_WIDTH 1000

using namespace std;
using namespace components;

namespace operators
{

//! HW 29/03/15 : To generate cuRAND pseudorundom numbers
#ifdef CUDA_CODE

KERNEL void K_seedSetup(curandState *state, unsigned int seed, int w, int h)
{
    KER_SCHED(w, h)

    if (_x < w && _y < h) {
        int cid = _x + _y * w;
        /* Each thread gets same seed, a different sequence
           number, no offset */
        curand_init(seed, cid, 0, &state[cid]);
        /* Each thread gets a different seed, a different sequence
           number, no offset */
//        curand_init(seed + (unsigned)(cid * 123456), cid, 0, &state[cid]);
    }
}

template<class Grid>
KERNEL void K_generateUniform(curandState *state,
                              Grid gRand)
{
    KER_SCHED(gRand.getWidth(), gRand.getHeight())

    if (_x < gRand.getWidth() && _y < gRand.getHeight()) {
        int cid = _x + _y * gRand.getWidth();
        /* Copy state to local memory for efficiency */
        curandState localState = state[cid];
        /* Generate pseudo-random uniforms */
        for(int i = 0; i < MAX_RAND_BUFFER_SIZE; i++) {
            gRand[_y][_x][i] = curand_uniform(&localState);
        }
        /* Copy state back to global memory */
        state[cid] = localState;
    }
}
#endif

template <class Cell>
struct GetAdaptor {//std adaptor
    DEVICE_HOST inline void init(int n) = 0;
    DEVICE_HOST inline void init(Cell& cell) = 0;

    //! HW 29/03/15 : method overloading for calling from device
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) = 0;

    DEVICE_HOST inline bool next(Cell& cell) = 0;
};

template <class Cell>
struct GetStdAdaptor {//std adaptor

    size_t nb;
    size_t niter;

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        cell.init();
    }

    //! HW 29/03/15 : method overloading for calling from device
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        return cell.get();
    }

    DEVICE_HOST inline bool next(Cell& cell) {
        return cell.next();
    }
};

template <class Cell>
struct GetRandomAdaptor {

    //!JCC 300315 : put device_host default
    DEVICE_HOST GetRandomAdaptor() : nb(), niter(1) {
#ifdef CUDA_CODE
#else
        aleat_initialize();
#endif
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
#ifdef CUDA_CODE
#else
        aleat_initialize();
#endif
    }

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    size_t nb;
    size_t niter;

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

#ifdef CUDA_CODE
    DEVICE inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        GLfloat rand = 0.0f;
        //! HW 29/03/15 : to generate cuRAND pseudorundom numbers using CUDA device API
        //! Note that this function is inefficient because each time a new seed (state)
        //! is set up by curand_init().
        //! Note that calls to curand_init() are slower than calls to curand() or curand_uniform().
        //! Therefore, it is much faster to save and restore random generator state
        //! than to recalculate the starting state repeatedly.
        //! Read more at: http://docs.nvidia.com/cuda/curand/
        curandState devStates;
        //! JCC 300315 : remove code in comment
        //! - response : This comment code maybe useful because it differs the seed of each thread/cell.
        //! - response : The current code uses the same seed for each thread/cell but with different sequence number.
        curand_init((unsigned long long)(clock()),
                    (cell.PC[0] + cell.PC[1] * MAX_CM_WIDTH),
                    0,
                    &devStates);
//        curand_init(((unsigned long long)(clock()) + (unsigned long long)((cell.PC[0] + cell.PC[1] * MAX_CM_WIDTH) * 123456)),
//                    (cell.PC[0] + cell.PC[1] * MAX_CM_WIDTH),
//                    0,
//                    &devStates);
        rand = curand_uniform(&devStates);
        return cell.extractRandom(nn, ps, rand);
    }
#else
    //! JCC a host only function doesn't need be called HOST
    inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        GLfloat rand = aleat_float(0, 1);
        return cell.extractRandom(nn, ps, rand);
    }
#endif

    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < niter;
    }
};

//! JCC 310315 : I create specific RandGridAllocator
//! HW 29/03/15 : add typedef and enum
typedef components::Point<GLfloat, MAX_RAND_BUFFER_SIZE> RandNumberBuffer;
typedef Grid<RandNumberBuffer> RandGrid;

//!
//! \brief The RandGridAlloc struct
//!
struct RandGridAlloc {
    unsigned int seed;

    DEVICE_HOST RandGridAlloc() {
        seed = time(NULL);
#ifdef CUDA_CODE
#else
        aleat_initialize(seed);
#endif
    }

    GLOBAL void generateRandNumBuffer(RandGrid gRand) {

        //! JCC changing parameter passing
        size_t w = gRand.getWidth();
        size_t h = gRand.getHeight();

#ifdef CUDA_CODE
        /* Allocate space for prng states on device */
        curandState *devStates;
        cudaMalloc((void **)&devStates, w * h * sizeof(curandState));

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              w, h);
        K_seedSetup _KER_CALL_(b, t) (devStates,
                                      seed,
                                      w, h);
        K_generateUniform<Grid<RandNumberBuffer> >  _KER_CALL_(b, t) (devStates,
                                                                      gRand);
#else
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
                for (int k = 0; k < MAX_RAND_BUFFER_SIZE; ++k)
                {
                    gRand[j][i][k] = aleat_float(0, 1);
                }
#endif
    }
};

//!
//! \brief The RandGridAllocFromCPU struct
//!
struct RandGridAllocFromCPU {

    RandGridAllocFromCPU() {
        aleat_initialize();
    }

    //! JCC 310315 : no "GLOBAL" since only CPU possible
    void generateRandNumBuffer(RandGrid gRand) {

        size_t w = gRand.getWidth();
        size_t h = gRand.getHeight();
        RandGrid gRand_cpu(w, h);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
                for (int k = 0; k < MAX_RAND_BUFFER_SIZE; ++k)
                {
                    gRand_cpu[j][i][k] = aleat_float(0, 1);
                }

        gRand_cpu.gpuCopyHostToDevice(gRand);
        //! JCC 300315 : free tmp memory
        gRand_cpu.freeMem();
    }
};

//! JCC 300315 : dissociation from Random allocation
//! HW 29/03/15 : to generate two grids of rundom numbers
template <class Cell>
struct GetRandomGridAdaptor {

    //! Random numbers
    RandGrid gRandInitiator;
    RandGrid gRandRoulette;

    //! max density cell
    GLfloat max_cell_density;
    GetRandomGridAdaptor() : nb(0), niter(1) {}

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
        size_t w = cm.getWidth();
        size_t h = cm.getHeight();
        gRandInitiator.gpuResize(w, h);
        gRandRoulette.gpuResize(w, h);
        RandGridAlloc rga;
        rga.generateRandNumBuffer(gRandInitiator);
        rga.generateRandNumBuffer(gRandRoulette);
        max_cell_density = cm.getMaxCellDensity();
    }

    size_t nb;
    size_t niter;

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
        nb = 0;
        niter = n;
    }

    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = false;
        GLfloat rand1 = 0.0f;
        rand1 = gRandInitiator[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
        // Cell activation
        if (cell.density >= rand1 * max_cell_density) {
            GLfloat rand2 = 0.0f;
            rand2 = gRandRoulette[cell.PC[1]][cell.PC[0]][(niter*gene+nb) % MAX_RAND_BUFFER_SIZE];
            // Roulette wheel random extraction
            ret = cell.extractRandom(nn, ps, rand2);
        }
        return ret;
    }

    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < niter;
    }
};

template <class Cell>
struct GetDivideAdaptor {
    size_t size_slab;
    size_t curPos;

    GetDivideAdaptor() {}

    DEVICE_HOST GetDivideAdaptor(size_t w, size_t h, size_t W, size_t H) {
        size_t snet = w * h;
        size_t sdiv = W * H;

        size_slab = (snet + sdiv - 1) / sdiv;
        curPos = 0;
    }

    template <class CellularMatrix>
    void initialize(CellularMatrix& cm) {
    }

    size_t gene;
    DEVICE_HOST inline void setGene(int n) {
        gene = n;
    }

    DEVICE_HOST inline void init(int n) {
    }

    DEVICE_HOST inline void init(Cell& cell) {
        curPos = 0;
    }

    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        bool ret = curPos < size_slab;
        if (ret) {
            size_t nnidx = cell.PC[1] * cell.vgd.getWidthDual() * size_slab
                    + cell.PC[0]*size_slab + curPos;
            ps[1] = nnidx / nn.adaptiveMap.getWidth();
            ps[0] = nnidx % nn.adaptiveMap.getWidth();
        }
        return ret;
    }

    DEVICE_HOST inline bool next(Cell cell) {
        return ++curPos < size_slab;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchAdaptor {

    DEVICE_HOST virtual bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) = 0;
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchIdAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {
        minP = ps;
        return true;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchCenterAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {
        minP = cm.vgd.FDual(PC);
        return true;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchCenterToCenterAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {
        minP = cm.vgd.FDual(PC);
        ps = minP;
        return true;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchSpiralAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord PC,
                            PointCoord ps,
                            PointCoord& minP) {
        bool ret = false;

        SpiralSearchCMIterator sps_iter(
                    PC,
                cm.vgd.getWidth() * cm.vgd.getHeight(),
                0,
                cm.vgd.getHeightDual(),
                2
                );
        ret = sps_iter.search(cm,
                              scher,
                              sched,
                              ps,
                              minP);
        return ret;
    }
};

template <class CellularMatrix,
          class SpiralSearchCMIterator>
struct SearchFindCellAdaptor {

    DEVICE_HOST bool search(CellularMatrix& cm,
                            NN& scher,
                            NN& sched,
                            PointCoord& PC,
                            PointCoord& ps,
                            PointCoord& minP) {

        if (ps[0] < sched.adaptiveMap.getWidth() &&
                ps[1] < sched.adaptiveMap.getHeight())
        minP = cm.vgd.findCell(sched.adaptiveMap[ps[1]][ps[0]]);
        return true;
    }
};

template <class Cell>
struct OperateAdaptor {
    DEVICE_HOST virtual void operate(Cell& cell, NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) = 0;
};

static DEVICE_HOST GLfloat chap(GLfloat d, GLfloat rayon)
{
    return(exp(-(d * d)/(rayon * rayon)));
}

template <class Cell,
          class NIter>
struct OperateTriggerAdaptor {

    GLfloat alpha;
    GLfloat radius;

    DEVICE_HOST OperateTriggerAdaptor() : alpha(), radius(){}

    DEVICE_HOST OperateTriggerAdaptor(GLfloat a, GLfloat r) : alpha(a), radius(r){}

    DEVICE_HOST void operate(Cell& cell,  NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        if (p_source[0] >= 0 && p_source[0] < nn_source.adaptiveMap.getWidth()
                && p_source[1] >= 0 && p_source[1] < nn_source.adaptiveMap.getHeight()) {
        PointEuclid p = nn_source.adaptiveMap[p_source[1]][p_source[0]];
        NIter ni(p_cible, 0, radius);

        for (int d = 0; d <= radius; ++d) {

            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            GLfloat alpha_temp = alpha * chap((GLfloat)cd, (GLfloat)radius*LARG_CHAP);
            PointCoord pCoord;
            do {

                pCoord = ni.getNodeIncr();
                if (pCoord[0] >= 0 && pCoord[0] < nn_cible.adaptiveMap.getWidth()
                        && pCoord[1] >= 0 && pCoord[1] < nn_cible.adaptiveMap.getHeight()) {

                    if (!nn_cible.fixedMap[pCoord[1]][pCoord[0]]) {
                        PointEuclid n = nn_cible.adaptiveMap[pCoord[1]][pCoord[0]];
                        n[0] = n[0] + alpha_temp * (p[0] - n[0]);
                        n[1] = n[1] + alpha_temp * (p[1] - n[1]);
                        nn_cible.adaptiveMap[pCoord[1]][pCoord[0]] = n;
#if SOM3D
                        nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * 1.0;
                        //nn_cible.densityMap[pCoord[1]][pCoord[0]] += alpha_temp * (nn_cible.densityMap[pCoord[1]][pCoord[0]] - nn_source.densityMap[pCoord[1]][pCoord[0]]);
#endif
                    }

                }
            } while (ni.nextContourNodeIncr());
        }//for
        }//if
    }//operate
};

template <template<typename, typename> class NeuralNet, class Cell>
struct OperateInsertAdaptor {

    DEVICE_HOST OperateInsertAdaptor() {}

    DEVICE_HOST void operate(Cell& cell,  NN& nn_source, NeuralNet<Cell, GLfloat>& nn_cible, PointCoord p_source, PointCoord p_cible) {

        Cell ci = (Cell) nn_cible.adaptiveMap[p_cible[1]][p_cible[0]];
        ci.insert(p_source);

    }//operateInsert
};

}//namespace operators

#endif // ADAPTATOR_BASICS_H
