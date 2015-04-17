#ifndef SPIRAL_SEARCH_H
#define SPIRAL_SEARCH_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput
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


#include "NIter.h"


#include "distances_matching.h"
#include "Cell.h"

#define TEST_CODE 0

using namespace std;

namespace components
{

/*!
 * \brief The SpiralSearchIterator struct
 */

template <
          class Cell,
          class NIter
         >
struct SpiralSearchCMIterator {

    size_t start;
    size_t d_min;
    size_t d_max;
    size_t d_step;

    PointCoord pc;//Coordinates cell center in gdc

    GLfloat minDistance;

    DEVICE_HOST SpiralSearchCMIterator(
            PointCoord pc,
            GLfloat md,
            size_t d_min,
            size_t d_max,
            size_t d_step
            )
        :
            pc(pc),
            minDistance(md),
            start(0),
            d_min(d_min),
            d_max(d_max),
            d_step(d_step)
    {}

    DEVICE_HOST bool search(Grid<Cell>& gdc, NN& scher, NN& sched, PointCoord ps, PointCoord& minPCoord) {

        bool ret = false;
        NIter ni(pc, 0, d_max);//first if no finding

        do {
            if (ni.getCurrentDistance() > d_max)
                break;

            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < gdc.getWidth()
                    && pco[1] >= 0 && pco[1] < gdc.getHeight()) {

                Cell cell = gdc[pco[1]][pco[0]];

                if (cell.search(scher, sched, ps)) {
                    if (cell.getMinDistance() < minDistance)
                    {
                        ret = true;

                        // Change max
                        ni.setMaxDistance(max(d_min, ni.getCurrentDistance()+d_step));

                        minDistance = cell.getMinDistance();
                        minPCoord = cell.getMinPCoord();
                    }
                }
            }
        } while (ni.nextNodeIncr());

        return ret;
    }

};//SpiralSearchCMIterator

template <
          class Distance,
          class Condition,
          class NIter
        >
struct SpiralSearchNNIterator {
    size_t start;
    size_t d_min;
    size_t d_max;
    size_t d_step;

    PointCoord pc;//Coordinates cell center in gdc

    GLfloat minDistance;

    DEVICE_HOST SpiralSearchNNIterator(
            PointCoord pc,
            GLfloat md,
            size_t d_min,
            size_t d_max,
            size_t d_step
            )
        :
            pc(pc),
            minDistance(md),
            start(0),
            d_min(d_min),
            d_max(d_max),
            d_step(d_step)
    {}

    //! Search function at coordinate level
    DEVICE_HOST bool search(NN& scher, NN& sched, PointCoord ps, PointCoord& minPCoord) {

        bool ret = false;
        NIter ni(pc, 0, d_max);//first if no finding

        do {
            if (ni.getCurrentDistance() > d_max)
                break;

            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < sched.adaptiveMap.getWidth()
                    && pco[1] >= 0 && pco[1] < sched.adaptiveMap.getHeight()) {

                Distance dist;
                Condition cond;
                GLfloat v = dist(ps, pco, scher, sched);
                bool c = cond(ps, pco, scher, sched);
                if (v < minDistance && c)
                {
                    ret = true;

                    // Change max
                    ni.setMaxDistance(max(d_min, ni.getCurrentDistance()+d_step));

                    minDistance = v;
                    minPCoord = pco;
                }
            }
        } while (ni.nextNodeIncr());

        return ret;
    }

};//SpiralSearchNNIterator


}//namespace components

#endif // SPIRAL_SEARCH_H
