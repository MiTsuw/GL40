#ifndef TESTCELLULAR_H
#define TESTCELLULAR_H
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
#include <cuda_runtime.h>
#include <cuda.h>
//#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>
#endif

#include "macros_cuda.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "distances_matching.h"
#include "Cell.h"
#include "SpiralSearch.h"
#include "Objectives.h"
#include "Trace.h"
#include "ConfigParams.h"
#include "CellularMatrix.h"
#include "ImageRW.h"

#include "SpiralSearch.h"

#include "ViewGrid.h"
//#ifdef TOPOLOGIE_HEXA
//#include "ViewGridHexa.h"
//#endif
#include "NIter.h"
//#ifdef TOPOLOGIE_HEXA
//#include "NIterHexa.h"
//#endif

using namespace std;
using namespace components;
using namespace operators;

namespace meshing
{

/*!
 * \brief The TestCellular class
 */
class TestCellular {
    char* fileData;
    char* fileSolution;
    char* fileStats;
    ConfigParams params;
    Trace trace;

public:
    // Types

    typedef ViewGridQuad ViewG;

    typedef CellB<CM_DistanceEuclidean,
                    CM_ConditionTrue,
                    NIterQuad, ViewG> CB;
    typedef CellSpS<CM_DistanceEuclidean,
                    CM_ConditionTrue,
                    NIterQuad, ViewG> CSpS;

    typedef CellularMatrix<CSpS, ViewG> CMSpS;
    typedef CellularMatrix<CB, ViewG> CMB;

    // Data
    NN md;
    NN mr;
    NN md_gpu;
    NN mr_gpu;
    ViewG vgd;
    CMSpS cmd;
    CMSpS cmr;

    // Adaptator which needs initialize()
    GetRandomGridAdaptor<CSpS> ga;

    TestCellular(char* fileData, char* fileSolution, char* fileStats, ConfigParams params) :
        fileData(fileData),
        fileSolution(fileSolution),
        fileStats(fileStats),
        params(params), vgd()
    {}

    void initialize() {
        // Data

        // Initialize NN netwoks CPU/GPU
        // Image read/write
        IRW irw;
        // Matched
        irw.read(fileData, md);
        size_t _w = md.colorMap.getWidth();
        size_t _h = md.colorMap.getHeight();
        md_gpu.gpuResize(_w, _h);
        md.adaptiveMap.resize(_w, _h);
        md_gpu.adaptiveMap.gpuResize(_w, _h);
        md.gpuCopyHostToDevice(md_gpu);

        // Matcher
        irw.read(fileData, mr);
        mr_gpu.gpuResize(_w, _h);
        mr.adaptiveMap.resize(_w, _h);
        mr.fixedMap.resize(_w, _h);
        mr.fixedMap = false;
        mr.densityMap.resize(_w, _h);
        mr_gpu.adaptiveMap.gpuResize(_w, _h);
        mr_gpu.fixedMap.gpuResize(_w, _h);
        mr_gpu.densityMap.gpuResize(_w, _h);
        mr.gpuCopyHostToDevice(mr_gpu);

        // ViewGrid
        PointCoord pc(_w / 2, _h / 2 + 1);
        int _R = params.levelRadius;
        vgd = ViewG(pc, _w, _h);

        cout << "vgd dual " << vgd.getWidthDual() << " "
                << vgd.getHeightDual() << endl;

        // Cellular matrix matched
        cmd.setViewG(vgd);
        cmd.gpuResize(vgd.getWidthDual(), vgd.getHeightDual());
        cmd.K_initialize(vgd);
        cmd.K_cellDensityComputation(md_gpu);
        cmd.K_initializeRegularIntoPlane<LOW_LEVEL>(vgd, md_gpu.adaptiveMap);

        // Cellular matrix matcher
        cmr.setViewG(vgd);
        cmr.gpuResize(vgd.getWidthDual(), vgd.getHeightDual());
        cmr.K_initialize(vgd);
        cmr.K_cellDensityComputation(mr_gpu);
        cmr.K_initializeRegularIntoPlane<LOW_LEVEL>(vgd, mr_gpu.adaptiveMap);

        // Trace object
        trace.initialize(fileStats);

        ga.initialize(cmd);
        ga.init(100);

    }//initialize()

    void run() {

        // Projection adaptators
        //GetRandomAdaptor<CSpS> ga;
       // string file_cfg;
       // param->readConfigParameter("param_1","fileGrid3DPoints", file_cfg);


        int testCellular;

        params.readConfigParameter("test_cellular","testCellular", testCellular);

        if (testCellular == 0) {
            SearchCenterToCenterAdaptor<CMSpS, NIterQuadDual> sa;
            OperateTriggerAdaptor<CSpS, NIterQuad> oa(1.0, vgd.getLevel()-1);
            // Projection
            cmd.K_projector(cmr, md_gpu, mr_gpu, ga, sa, oa);
        }
        else if (testCellular == 1) {
            SearchCenterAdaptor<CMSpS, NIterQuadDual> sa;
            OperateTriggerAdaptor<CSpS, NIterQuad> oa(1.0, vgd.getLevel()-1);
            // Projection
            cmd.K_projector(cmr, md_gpu, mr_gpu, ga, sa, oa);
        }
        else {
            SearchIdAdaptor<CMSpS, NIterQuadDual> sa;
            OperateTriggerAdaptor<CSpS, NIterQuad> oa(1.0, vgd.getLevel()-1);
            // Projection
            cmd.K_projector(cmr, md_gpu, mr_gpu, ga, sa, oa);
        }

        // Get from device
        //!JCC : md.gpuCopyDeviceToHost(md_gpu);
        mr.gpuCopyDeviceToHost(mr_gpu);

        // Objectives
        AMObjectives objs;

        class Evaluation {
        public:
            void evaluate(AMObjectives& objs) {}
        };

        Evaluation eval;//(/*cmr, md_gpu, mr_gpu, ga, sa, oa*/);
        eval.evaluate(objs);

        // Trace Writting
        trace.setObjs(objs);
        trace.writeStatistics(0, cout);
        trace.writeStatistics(0);
        trace.closeStatistics();

        // Save solution
        mr.write(fileSolution);
    }
};

}//namespace meshing

#endif // TESTCELLULAR_H
