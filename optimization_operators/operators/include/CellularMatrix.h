#ifndef CELLULAR_MATRIX_H
#define CELLULAR_MATRIX_H
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
#ifdef CUDA_CODE
#include <cuda_runtime.h>"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

//#include <helper_functions.h>


#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"
#include "distance_functors.h"
#include "adaptator_basics.h"

#include "SpiralSearch.h"

#include "ViewGrid.h"
#include "NIter.h"


using namespace components;
using namespace std;

namespace operators
{

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
            //! HW 29/03/15 : modif
            //! JCC 300315 : adaptator is overload for Som only
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
    GLfloat maxCellDensity;

    DEVICE_HOST CellularMatrix() : maxCellDensity(1) {}
    DEVICE_HOST CellularMatrix(ViewG viewgd)
        : maxCellDensity(1), vgd(viewgd) {}

    DEVICE_HOST void setViewG(ViewG& v) { vgd = v; }
    DEVICE_HOST ViewG& getViewG() { return vgd; }
    DEVICE_HOST void setMaxCellDensity(GLfloat m) { maxCellDensity = m; }
    DEVICE_HOST GLfloat getMaxCellDensity() { return maxCellDensity; }

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

        //! HW 01/04/15 : modif
        //! Otherwise GPU verison will crash because here (*this)._data now
        //! points to GPU device memory, and can not be directly accessed on host side.
        CellularMatrix tmp;
        tmp.resize(this->getWidth(), this->getHeight());
        tmp.gpuCopyDeviceToHost(*this);
        this->maxCellDensity = 1;
        for (int y = 0; y < this->getHeight(); y++) {
            for (int x = 0; x < this->getWidth(); x++)
            {
                if (tmp[y][x].density >= this->maxCellDensity)
                {
                    this->maxCellDensity = tmp[y][x].density;
                }
            }
        }
        tmp.freeMem();
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
    GLOBAL inline void K_projector(CellularMatrix& cible, NN& nn_source, NN& nn_cible,  GetAdaptor& ga,   SearchAdaptor& sa,  OperateAdaptor& oa) {

        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              this->getWidth(),
                              this->getHeight());
        K_CM_projector _KER_CALL_(b, t) (
                            *this, cible, nn_source, nn_cible, ga, sa, oa);
    }

}; // class CellularMatrix

}//namespace operators

#endif // CELLULAR_MATRIX_H
