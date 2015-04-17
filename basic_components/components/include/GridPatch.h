#ifndef GRIDPATCH_H
#define GRIDPATCH_H
/*
 ***************************************************************************
 *
 * Author : H. Wang, J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
#include <boost/geometry/geometries/point.hpp>

#include "lib_global.h"
#include "GridOfNodes.h"


#include "NIter.h"
#ifdef TOPOLOGIE_HEXA
#include "NIterHexa.h"
#endif


using namespace std;

namespace components
{

/*!
 * \defgroup Grid
 * \brief Espace de nommage components
 * Il comporte les patchs
 */
/*! @{*/

template <class Node, class NIter>
class GridPatch : public Grid<Node> {

    size_t radius;
    PointCoord pc; //grid center
    PointCoord ppc; //related patch center

public:    

    //! Default constructor (do not access)
    DEVICE_HOST inline GridPatch() : Grid<Node>(),
        radius(0),
        pc(0,0),
        ppc(0,0) {}

    DEVICE_HOST explicit inline GridPatch(size_t r) :
        radius(r),
        pc(0,0),
        ppc(0,0) {}

    DEVICE_HOST explicit GridPatch(size_t r, PointCoord p, Grid<Node> gd) :
        radius(r),
        pc(p) {

        //! Get related point
        ppc = gridToPatch(pc);

        //! Allocation only once
        if ((pc[1] - radius) % 2 == 0)
            this->resize(2*r+1, 2*r+1);
        else
            this->resize(2*r+1, 2*r+2);

        //! Copy patch
        extractPatchFromGrid(gd);

    } //constructor

    DEVICE_HOST inline size_t getRadius() {
        return radius;
    }

    DEVICE_HOST inline PointCoord getGridCenter() {
        return pc;
    }

    DEVICE_HOST inline PointCoord getPatchCenter() {
        return ppc;
    }

    DEVICE_HOST inline PointCoord patchToGrid(PointCoord & pp)
    {
        PointCoord pg;
        if ((pc[1] - radius) % 2 == 0)
        {
            pg[0] = pp[0] + (pc[0] - radius);
            pg[1] = pp[1] + (pc[1] - radius);
        }
        else
        {
            pg[0] = pp[0] + (pc[0] - radius);
            pg[1] = pp[1] + (pc[1] - radius - 1);
        }
        return pg;
    }

    DEVICE_HOST inline PointCoord gridToPatch(PointCoord & pg)
    {
        PointCoord pp;
        if ((pc[1] - radius) % 2 == 0)
        {
#if TEST_CODE
            if (pg[0] >= (pc[0] - radius)
                    && pg[0] <= (pc[0] + radius)
                    && pg[1] >= (pc[1] - radius)
                    && pg[1] <= (pc[1] + radius))
            {
#endif
                pp[0] = pg[0] - (pc[0] - radius);
                pp[1] = pg[1] - (pc[1] - radius);
#if TEST_CODE
            }
            else
            {
                printf("The coordinate is out of the patch !\n");
            }
#endif
        }
        else
        {
#if TEST_CODE
            if (pg[0] >= (pc[0] - radius)
                    && pg[0] <= (pc[0] + radius)
                    && pg[1] >= (pc[1] - radius - 1)
                    && pg[1] <= (pc[1] + radius))
            {
#endif
                pp[0] = pg[0] - (pc[0] - radius);
                pp[1] = pg[1] - (pc[1] - radius - 1);
#if TEST_CODE
            }
            else
            {
                printf("The coordinate is out of the patch !\n");
            }
#endif
        }
        return pp;
    }

    DEVICE_HOST void readPatchFromGrid(Grid<Node> & gd)
    {
        NIter ni(pc, 0, radius);
        PointCoord pg;
        do {
            pg = ni.getNodeIncr();
            if (pg[0] >= 0 && pg[0] < gd.getWidth()
                    && pg[1] >= 0 && pg[1] < gd.getHeight())
            {
                PointCoord pp;
                pp = gridToPatch(pg);
                (*this)[pp[1]][pp[0]] = gd[pg[1]][pg[0]];
            }
        } while (ni.nextNode());
    }

    DEVICE_HOST void writePatchToGrid(Grid<Node> & gd)
    {
        NIter ni(pc, 0, radius);
        PointCoord pg;
        do {
            pg = ni.getNodeIncr();
            if (pg[0] >= 0 && pg[0] < gd.getWidth()
                    && pg[1] >= 0 && pg[1] < gd.getHeight())
            {
                PointCoord pp;
                pp = gridToPatch(pg);
                gd[pg[1]][pg[0]] = (*this)[pp[1]][pp[0]] ;
            }
        } while (ni.nextNode());
    }

    //! The faster way
    DEVICE_HOST void extractPatchFromGrid(Grid<Node> & gd)
    {
        PointCoord pg;
        for (int y = -((int)radius); y <= ((int)radius); ++y)
        {
            for (int x = -((int)radius); x <= ((int)radius); ++x)
            {
                pg = PointCoord(pc[0]+x, pc[1]+y);
                if (pg[0] >= 0 && pg[0] < gd.getWidth()
                        && pg[1] >= 0 && pg[1] < gd.getHeight())
                {
                    (*this)[ppc[1]+y][ppc[0]+x] = gd[pg[1]][pg[0]];
                }
            }
        }
    }

    //! The faster way
    DEVICE_HOST void updateGridFromPatch(Grid<Node> & gd)
    {
        PointCoord pg;
        for (int y = -((int)radius); y <= ((int)radius); ++y)
        {
            for (int x = -((int)radius); x <= ((int)radius); ++x )
            {
                pg = PointCoord(pc[0]+x, pc[1]+y);
                if (pg[0] >= 0 && pg[0] < gd.getWidth()
                        && pg[1] >= 0 && pg[1] < gd.getHeight())
                {
                    gd[pg[1]][pg[0]] = (*this)[ppc[1]+y][ppc[0]+x];
                }
            }
        }
    }

}; // class GridPatch : public Grid<Node>

//! KERNEL FUNCTION
template<class Grid, class Node>
KERNEL void K_GridPatchPrint(Grid g1, size_t radius)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

//    if ((_x == g1.getWidth()/2 && _y == g1.getHeight()/2) || (_x < 3 && _y < 3) || (_x > g1.getWidth() - 3 && _y > g1.getHeight() - 3))
    if (_x == g1.getWidth()/2 && _y == g1.getHeight()/2)
//    if (_x == 1 && _y == 1)
//    if ((_x == g1.getWidth() - 1) && (_y == g1.getHeight() - 1))
    {
        PointCoord pCoord(_x,_y);
        GridPatch<Node, NIterQuad> gpatch(radius, pCoord, g1);
        printf("gpatch (%d,%d):\n", _x, _y);
        gpatch.printInt();

        PointCoord p1(_x - 1, _y + 2);
        PointCoord p11 = gpatch.gridToPatch(p1);
        printf("p11: gpatch.gridToPatch((%d,%d)) \n",_x - 1, _y + 2);
        p11.printInt();
        printf("\n");

        PointCoord p2(1,3);
        PointCoord p22 = gpatch.patchToGrid(p2);
        printf("p12: gpatch.patchToGrid((%d,%d)) \n", 1, 3);
        p22.printInt();
        printf("\n");

        Grid gg(g1.getWidth(), g1.getHeight());
        gg = Node(0);
        gpatch.updateGridFromPatch(gg);
        printf("gg (%d,%d):\n", _x, _y);
        gg.printInt();
        printf("\n");

        gg = Node(-1);
        gpatch.writePatchToGrid(gg);
        printf("gg (%d,%d):\n", _x, _y);
        gg.printInt();
        printf("\n");

        GridPatch<Node, NIterQuad> gpatch2(radius+1, pCoord, g1);
        gpatch2.readPatchFromGrid(g1);
        printf("gpatch2 (%d,%d):\n", _x, _y);
        gpatch2.printInt();
        printf("\n");
    }

    END_KER_SCHED

    SYNCTHREADS;
}

// Hongjian's Debug
//! Test program
template <class Grid, class Node>
class Test_GridPatch {

    Grid initGrid;

public:

    Test_GridPatch(Grid g1) : initGrid(g1) {}

    void run() {
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
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        }

        cout << "debut test Grid Patch ..." << endl;
        const size_t SX = initGrid.getWidth(), SY = initGrid.getHeight();
        // Creation de grille en local
        Grid gd(SX, SY);
        gd = initGrid;

        cout << "Creation de grilles sur device GPU ..." << endl;
        // Creation de grilles sur device GPU
        Grid gpu_gd;
        gpu_gd.gpuResize(SX, SY);

        // cuda timer
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
        double x0;
        x0 = clock();

        // Affichage

        (ofstream&) std::cout << "gd = " << endl;
        (ofstream&) std::cout << gd << endl;

        cout << "Appel du Kernel ..." << endl;

        // Copie des grilles CPU -> GPU
        gd.gpuCopyHostToDevice(gpu_gd);

        // Kernel call with class parameters
        KER_CALL_THREAD_BLOCK(b, t,
                              4, 4,
                              gpu_gd.getWidth(),
                              gpu_gd.getHeight());

        K_GridPatchPrint<Grid, Node> _KER_CALL_(b, t) (gpu_gd, 2);

        // Copie du resultat GPU -> CPU
        gd.gpuCopyDeviceToHost(gpu_gd);

        cout << "Affichage du resultat a la console ..." << endl;
        // Affichage du resultat à la console
        (ofstream&) std::cout << "gd_gpu = " << endl;
        (ofstream&) cout << gd << endl;

        // CPU Host test
        cout << "Affichage du resultat a la console (CPU host version) ..." << endl;
        int _x = gd.getWidth()/2;
        int _y = gd.getHeight()/2;
//        int _x = 1;
//        int _y = 1;

        PointCoord pCoord(_x,_y);
        GridPatch<Node, NIterQuad> gpatch(2, pCoord, gd);
        printf("gpatch (%d,%d):\n", _x, _y);
        gpatch.printInt();

        PointCoord p1(_x - 1, _y + 2);
        PointCoord p11 = gpatch.gridToPatch(p1);
        printf("p11: gpatch.gridToPatch((%d,%d)) \n",_x - 1, _y + 2);
        p11.printInt();
        printf("\n");

        PointCoord p2(1,3);
        PointCoord p22 = gpatch.patchToGrid(p2);
        printf("p12: gpatch.patchToGrid((%d,%d)) \n", 1, 3);
        p22.printInt();
        printf("\n");

        Grid gg(gd.getWidth(), gd.getHeight());
        gg = Node(0);
        gpatch.updateGridFromPatch(gg);
        printf("gg (%d,%d):\n", _x, _y);
        gg.printInt();
        printf("\n");

        gg = Node(-1);
        gpatch.writePatchToGrid(gg);
        printf("gg (%d,%d):\n", _x, _y);
        gg.printInt();
        printf("\n");

        GridPatch<Node, NIterQuad> gpatch2(2+1, pCoord, gd);
        gpatch2.readPatchFromGrid(gd);
        printf("gpatch2 (%d,%d):\n", _x, _y);
        gpatch2.printInt();
        printf("\n");

        // cpu timer
        cout << "CPU Time : " << (clock() - x0)/CLOCKS_PER_SEC << endl;

        // cuda timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
        cout << "GPU Execution Time: " <<  elapsedTime << " ms" << endl;
        cout << endl;

        // Explicit
        gpu_gd.gpuFreeMem();
        gd.freeMem();
    }
};

} // namespace components
//! @}
#endif // GRIDPATCH_H
