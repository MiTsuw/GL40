#ifndef DISTANCE_FUNCTORS_H
#define DISTANCE_FUNCTORS_H
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
#include "SpiralSearch.h"

#include "ViewGrid.h"

#include "NIter.h"


using namespace std;
using namespace components;

namespace operators
{

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

}//namespace operators

#endif // DISTANCE_FUNCTORS_H
