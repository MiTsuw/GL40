#ifndef CELL_H
#define CELL_H
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


#include "ViewGrid.h"
#include "NIter.h"



#define TEST_CODE 0

using namespace std;

//! HW 04/03/15: I do modification at line 34
//! It should be <= not <, so that round(1.5) = 2 not 1
#define round(x) ((fabs(ceil(x) - (x)) <= fabs(floor(x) - (x))) ? ceil(x) : floor(x))

#define MAX_CELL_SIZE 256

namespace components
{

template <class Node>
class Buffer : public Point<Node, MAX_CELL_SIZE> {};

/*!
 * \brief The Cell struct
 */

template <class Distance,
          class Condition,
          class NIter,
          class ViewGrid>
struct Cell : public PointEuclid {

    PointCoord PC;//in gd dual/cell level
    PointCoord pc;//in gd low level
    size_t radius;
    ViewGrid vgd;

    PointCoord minPCoord;//in cible grid
    GLfloat minDistance;

    size_t size;
    GLfloat density;

    NIter iter;

    DEVICE_HOST Cell() {}

    DEVICE_HOST Cell(
            PointCoord PC,
            PointCoord pc,
            size_t radius,
            ViewGrid vg
            )
        :
          PC(PC),
          pc(pc),
          radius(radius),
          vgd(vg),
          iter(pc, 0, radius)
    {}

    //! HW 28/03/15 : I remove the "virtual" keyword, otherwise GPU version will go wrong.
    DEVICE_HOST void initialize(
            PointCoord PPC,
            PointCoord ppc,
            size_t rradius,
            ViewGrid& vvg
            ) {
        PC = PPC;
        pc = ppc;
        radius = rradius;
        vgd = vvg;
        iter.initialize(pc, 0, radius);
        size = 0;
    }

    //! HW 28/03/15 : I comment the virtual method declarations, otherwise GPU version will go wrong.
#ifdef VIRTUAL
    // To iterate
    DEVICE_HOST virtual  void init() = 0;
    DEVICE_HOST virtual  bool get(PointCoord& ps) = 0;
    DEVICE_HOST virtual  bool next() = 0;

    // Get one point random
    DEVICE_HOST virtual bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat rand) = 0;

    // Search Closest point
    DEVICE_HOST virtual bool search(NN& scher, NN& sched, PointCoord ps) = 0;

    // Insertion in buffer
    DEVICE_HOST virtual bool insert(PointCoord& pc) = 0;

    // Utilities
    DEVICE_HOST virtual void computeDensity(NN& neuralNet) = 0;
#endif
    DEVICE_HOST inline PointCoord getMinPCoord() { return minPCoord; }
    DEVICE_HOST inline GLfloat getMinDistance() { return minDistance; }
    DEVICE_HOST inline void clearCell() { size = 0; }
};

/*!
 * \brief The CellB struct
 */
template <class Distance,
          class Condition,
          class NIter,
          class ViewG
          >
struct CellB : public Cell<Distance, Condition, NIter, ViewG> {

    Buffer<PointCoord> bCell;
    size_t curPos;

    DEVICE_HOST CellB() {}

    DEVICE_HOST CellB(
            PointCoord PC,
            PointCoord pc,
            size_t radius,
            ViewG vg
            )
        :
          Cell<Distance, Condition, NIter, ViewG>(
              PC,
              pc,
              radius,
              vg
              )
    {}

    // To iterate
    DEVICE_HOST inline void init() { curPos = 0; }
    DEVICE_HOST inline bool get(PointCoord& ps) {
        bool ret = curPos < this->size;
        if (ret)
            ps = bCell[curPos];
        return (ret);
    }
    DEVICE_HOST inline bool next() {
        return ++curPos < this->size;
    }

    //! From density map
    DEVICE_HOST bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat rand) {
        bool ret = false;
        size_t count = 0;
        GLfloat sum = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
            sum += value;
            if (sum > rand)
            {
                ret = true;
                ps = pco;
                break;
            }
            count++;
        }
        return ret;
    }//searchB

    //! Search coordinate level
    DEVICE_HOST bool search(NN& scher, NN& sched, PointCoord ps) {
        bool ret = false;
        size_t count = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            Distance dist;
            Condition cond;
            GLfloat v = dist(ps, pco, scher, sched);
            bool c = cond(ps, pco, scher, sched);
            if (v < this->minDistance && c)
            {
                ret = true;
                this->minDistance = v;
                this->minPCoord = pco;
            }
            count++;
        }
        return ret;
    }//searchB

    // Insertion in buffer
    DEVICE_HOST bool insert(PointCoord& pc) {
        bool ret = true;
#ifdef CUDA_CODE
#else
        if (this->size < MAX_CELL_SIZE) {
            this->size += 1;
            bCell[this->size] = pc;
        }
#endif
        return ret;
    }

    //! From density map
    DEVICE_HOST void computeDensity(NN& neuralNet) {
        size_t count = 0;
        GLfloat sum = 0;
        while (count < this->size)
        {
            PointCoord pco = bCell[count];

            GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
            sum += value;
            count++;
        }
        this->density = sum;
    }//searchB
};

/*!
 * \brief The CellSpS struct
 */
template <class Distance,
          class Condition,
          class NIter,
          class ViewG
          >
struct CellSpS : public Cell<Distance, Condition, NIter, ViewG> {

    DEVICE_HOST CellSpS() {}

    //! HW 09/03/15 : I modify at lines 184-191.
    DEVICE_HOST CellSpS(
            PointCoord PC,
            PointCoord pc,
            size_t radius,
            ViewG vg
            )
        :
          Cell<Distance, Condition, NIter, ViewG>(
              PC,
              pc,
              radius,
              vg
              )
    {}

    // To iterate
    DEVICE_HOST inline void init() { this->iter.init(); }
    DEVICE_HOST inline bool get(PointCoord& ps) {
        ps = this->iter.get();
        return (ps[0] >= 0 && ps[0] < this->vgd.getWidth()
                && ps[1] >= 0 && ps[1] < this->vgd.getHeight());
    }
    DEVICE_HOST inline bool next() { return this->iter.next(); }

    //! Euclidean/Value level
    DEVICE_HOST bool extractRandom(NN& neuralNet, PointCoord& ps, GLfloat random) {
        bool ret = false;
        NIter ni(this->pc, 0, this->radius);
        GLfloat sum = 0;
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < neuralNet.densityMap.getWidth()
                    && pco[1] >= 0 && pco[1] < neuralNet.densityMap.getHeight()) {
                GLfloat value = neuralNet.densityMap[pco[1]][pco[0]];
                sum += value;
                if (sum >= random * this->density)
                {
                    ret = true;
                    ps = pco;
                    break;
                }
            }
        } while (ni.nextNodeIncr());
        return ret;
    }

    //! Search function at coordinate level
    DEVICE_HOST bool search(NN& scher, NN& sched, PointCoord ps) {
        bool ret = true;
        NIter ni(this->pc, 0, this->radius);
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < sched.adaptiveMap.getWidth()
                    && pco[1] >= 0 && pco[1] < sched.adaptiveMap.getHeight()) {
                Distance dist;
                Condition cond;
                GLfloat v = dist(ps, pco, scher, sched);
                bool c = cond(ps, pco, scher, sched);
                if (v < this->minDistance && c)
                {
                    ret = true;
                    this->minDistance = v;
                    this->minPCoord = pco;
                }
            }
        } while (ni.nextNodeIncr());
        return ret;
    }

    // Insertion in buffer
    DEVICE_HOST bool insert(PointCoord& pc) {
        return true;
    }

    DEVICE_HOST void computeDensity(NN& nn) {
        NIter ni(this->pc, 0, this->radius);
        GLfloat sum = 0;
        do {
            PointCoord pco = ni.getNodeIncr();

            if (pco[0] >= 0 && pco[0] < nn.densityMap.getWidth()
                    && pco[1] >= 0 && pco[1] < nn.densityMap.getHeight()) {
                sum += nn.densityMap[pco[1]][pco[0]];
            }

        } while (ni.nextNodeIncr());
        this->density = sum;
    }
};

}//namespace components

#endif // CELL_H
