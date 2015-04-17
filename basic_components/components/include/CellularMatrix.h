#ifndef CELLULAR_MATRIX_H
#define CELLULAR_MATRIX_H
//***************************************************************************
//
// Jean-Charles Creput, 2013
//
//***************************************************************************
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

#include "random_generator.h"
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "NeighborhoodIterator.h"
#include "distances_matching.h"
#include "ViewGrid.h"
#include "Cell.h"
#include "SpiralSearch.h"

using namespace components;

namespace operators
{

#define LARG_CHAP 0.4f

struct CM_ConditionTrue
{
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return true;
    }
};

/*!
 * \brief The DistanceEuclidean struct
 * Basic functor for Euclidean distance
 */
struct CM_DistanceEuclidean
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        PointEuclid pp1 = nn1.adaptiveMap[p1[1]][p1[0]];
        PointEuclid pp2 = nn2.adaptiveMap[p2[1]][p2[0]];
        return components::DistanceEuclidean<PointEuclid>()(pp1, pp2);
    }
};

struct CM_ConditionNotFixed
{
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return !nn1.fixedMap[p1[1]][p1[0]] && !nn2.fixedMap[p2[1]][p2[0]];
    }
};

struct CM_ConditionActive
{
    // This operator is called for each segment
    DEVICE_HOST inline bool operator()(PointCoord& p1, PointCoord& p2, NN& nn1, NN& nn2)
    {
        return !nn1.activeMap[p1[1]][p1[0]] && !nn2.activeMap[p2[1]][p2[0]];
    }
};

//template<size_t R>
//class ViewG : public ViewGridHexa<R> {};

template <class CellularMatrix,
          class ViewG>
KERNEL void K_CM_initialize(CellularMatrix cm, ViewG vgd)
{
    KER_SCHED(cm.getWidth(), cm.getHeight())

    if (_x < cm.getWidth() && _y < cm.getHeight()) {
        PointCoord PC(_x,_y);
        PointCoord pc = vgd.FDual(PC);
        PointEuclid pe = vgd.FEuclid(pc);
        (PointEuclid&) cm[_y][_x] = pe;
        cm[_y][_x].initialize(PC, pc, vgd.getLevel(), vgd);
    }

    END_KER_SCHED

    SYNCTHREADS
}

template <class CellularMatrix>
KERNEL void K_CM_clearCells(CellularMatrix cm)
{
    KER_SCHED(cm.getWidth(), cm.getHeight())

    if (_x < cm.getWidth() && _y < cm.getHeight()) {
        cm[_y][_x].clearCell();
    }

    END_KER_SCHED

    SYNCTHREADS
}

template <class CellularMatrix>
KERNEL void K_CM_cellDensityComputation(CellularMatrix cm, NN nn)
{
    KER_SCHED(cm.getWidth(), cm.getHeight())

    if (_x < cm.getWidth() && _y < cm.getHeight()) {
        cm[_y][_x].computeDensity(nn);
    }

    END_KER_SCHED

    SYNCTHREADS
}

template <class CellularMatrix,
          class CellularMatrix2,
          class GetAdaptor,
          class SearchAdaptor,
          class OperateAdaptor
          >
KERNEL void K_CM_projector(CellularMatrix cm_source,
                           CellularMatrix2 cm_cible,
                           NN nn_source,
                           NN nn_cible,
                           GetAdaptor getAdaptor,
                           SearchAdaptor searchAdaptor,
                           OperateAdaptor operateAdaptor)
{
    KER_SCHED(cm_source.getWidth(), cm_source.getHeight())

    if (_x < cm_source.getWidth() && _y < cm_source.getHeight())
    {
        getAdaptor.init(cm_source[_y][_x]);
        do {
            PointCoord PC(_x, _y);

            // Partition
            PointCoord ps;
            bool extracted;
            extracted = getAdaptor.get(cm_source[_y][_x], nn_source, ps);

            if (extracted) {
                // Spiral search
                PointCoord minP;
                bool found;

                found = searchAdaptor.search(cm_cible,
                                             nn_source,
                                             nn_cible,
                                             PC,
                                             ps,
                                             minP);

                if (found) {
                    operateAdaptor.operate(cm_cible[_y][_x], nn_source, nn_cible, ps, minP);
                }
            }

        } while (getAdaptor.next(cm_source[_y][_x]));
    }

    END_KER_SCHED

    SYNCTHREADS
}

/*!
 * \brief The CellularMatrix class
 */

template <class Cell, class ViewG>
class CellularMatrix : public Grid<Cell>
{
public:

    ViewG vgd;
    DEVICE_HOST CellularMatrix(ViewG viewgd) : vgd(viewgd) {}

    //! HW 26.03.15 : modif
    //! "A template-parameter shall not be redeclared within its scope (including nested scopes).
    //!  A template-parameter shall not have the same name as the template name."
    //! Otherwise, the redeclared declaration shadows the formerly declared template parameter.
    template<class ViewG1>
    GLOBAL void K_initialize(ViewG1& vg) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_initialize _KER_CALL_(b, t) (*this, vg);
    }

    //! HW 26.03.15 : modif
    template<Level LEVEL, class ViewG1>
    GLOBAL inline void K_initializeRegularIntoPlane(ViewG1& vg, Grid<PointEuclid>& map) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              map.getWidth(),
                              map.getHeight());
        K_VG_initializeIntoPlane<ViewG1, Grid<PointEuclid>, LEVEL> _KER_CALL_(b, t) (
                    vg, map);
    }

    GLOBAL void K_clearCells() {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_clearCells _KER_CALL_(b, t) (*this);
    }

    GLOBAL void K_cellDensityComputation(NN& nn) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_cellDensityComputation _KER_CALL_(b, t) (*this, nn);
    }

    /*!
     * \brief K_projector
     *
     */
    template <class CellularMatrix,
              class GetAdaptor,
              class SearchAdaptor,
              class OperateAdaptor
              >
    GLOBAL inline void K_projector(CellularMatrix& cible, NN& nn_source, NN& nn_cible, class GetAdaptor& ga, class SearchAdaptor& sa, class OperateAdaptor& oa) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_projector _KER_CALL_(b, t) (
                            *this, cible, nn_source, nn_cible, ga, sa, oa);
    }

}; // class CellularMatrix

template <class Cell>
struct GetAdaptor {//std adaptor
    DEVICE_HOST inline void init(Cell& cell) = 0;
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) = 0;
    DEVICE_HOST inline bool next(Cell& cell) = 0;
};

template <class Cell>
struct GetStdAdaptor {//std adaptor
    DEVICE_HOST inline void init(Cell& cell) {
        cell.init();
    }
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        return cell.get();
    }
    DEVICE_HOST inline bool next(Cell& cell) {
        return cell.next();
    }
};

template <class Cell>
struct GetRandomAdaptor {

    GetRandomAdaptor() {
#ifdef CUDA_CODE
#else
        aleat_initialize();
#endif
    }

    size_t nb;
    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        GLfloat rand;
#ifdef CUDA_CODE
#else
        rand = aleat_float(0, 1);
#endif

        return cell.extractRandom(nn, ps, rand);
    }
    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < 1;
    }
};

template <class Cell>
struct GetRandomGridAdaptor {

    //! Random numbers
    Grid<GLfloat> gRandInitiator;
    Grid<GLfloat> gRandRoulette;

    GetRandomGridAdaptor() {
#ifdef CUDA_CODE
#else
        aleat_initialize();
#endif
    }

    size_t nb;
    DEVICE_HOST inline void init(Cell& cell) {
        nb = 0;
    }
    DEVICE_HOST inline bool get(Cell& cell, NN& nn, PointCoord& ps) {
        GLfloat rand;
#ifdef CUDA_CODE
#else
        rand = aleat_float(0, 1);
#endif

        return cell.extractRandom(nn, ps, rand);
    }
    DEVICE_HOST inline bool next(Cell& cell) {
        return ++nb < 1;
    }
};

template <class Cell>
struct GetDivideAdaptor {
    size_t size_slab;
    size_t curPos;

    DEVICE_HOST GetDivideAdaptor(size_t w, size_t h, size_t W, size_t H) {
        size_t snet = w * h;
        size_t sdiv = W * H;

        size_slab = (snet + sdiv - 1) / sdiv;
        curPos = 0;
    }
    DEVICE_HOST inline void init(const Cell& cell) {
        curPos = 0;
    }
    DEVICE_HOST inline bool get(const Cell& cell, const NN& nn, PointCoord& ps) {
        bool ret = curPos < size_slab;
        if (ret) {
            size_t nnidx = cell.PC[1]*cell.vgd.getWidthDual()*size_slab
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
        minP = cm.vgd.FDual(PC);
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
                cm[PC[1]][PC[0]].vgd.getWidth() * cm[PC[1]][PC[0]].vgd.getHeight(),
                0,
                cm[PC[1]][PC[0]].vgd.getHeightDual(),
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
    size_t rayon;

    DEVICE_HOST OperateTriggerAdaptor(GLfloat a, size_t r) : alpha(a), rayon(r){}

    DEVICE_HOST void operate(Cell& cell,  NN& nn_source, NN& nn_cible, PointCoord p_source, PointCoord p_cible) {

        PointEuclid p = nn_source.adaptiveMap[p_source[1]][p_source[0]];
        NIter ni(p_cible, 0, rayon);

        for (int d = 0; d <= rayon; ++d) {

            // Ready to next distance
            size_t cd = ni.getCurrentDistance();
            GLfloat alpha_temp = alpha * chap((GLfloat)cd, (GLfloat)rayon*LARG_CHAP);
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
                    }

                }
            } while (ni.nextContourNodeIncr());
        }
    }//operate
};


}//namespace operators


#endif // CELLULAR_MATRIX_H
