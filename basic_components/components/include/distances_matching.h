#ifndef DISTANCES_H
#define DISTANCES_H
/*
 ***************************************************************************
 *
 * Author : H. Wang, J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <vector>
#include <iterator>

#include "macros_cuda.h"
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"

#define TEST_CODE 0

using namespace std;
using namespace components;

namespace components
{

/*!
 * \defgroup Distances de matching
 * \brief Distances computed between nodes and windows.
 */
/*! @{*/

/*! @name Distances node to node.
 * \brief From a node to a node.
 */
//! @{
//! Distances node to node

//! Manhattan distance functor (L1 norm)
template<class Point>
struct DistanceManhattan
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(const Point& p1, const Point& p2)
    {
        GLfloat dist = fabs(p1 - p2);
        return dist;
    }
};

//! Euclidean distance functor (L2 norm)
template<class Point>
struct DistanceEuclidean
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(const Point& p1, const Point& p2)
    {
        GLfloat dist = (p1 - p2) * (p1 - p2);
        return sqrt(dist);
    }
};

//! Squared Euclidean distance functor
template<class Point>
struct DistanceSquaredEuclidean
{
    // This operator is called for each segment
    DEVICE_HOST inline GLfloat operator()(const Point& p1, const Point& p2)
    {
        GLfloat dist = (p1 - p2) * (p1 - p2);
        return dist;
    }
};

////! Distance tpye enumeration
//enum DistanceType {Manhattan, Euclidean, SquaredEuclidean};

////! Generic distance function
//template<class Point>
//DEVICE_HOST GLfloat distance(const Point& p1, const Point& p2, DistanceType dis_type)
//{
//    if (dis_type == Manhattan)
//    {
//        DistanceManhattan<Point> op;
//        return (op(p1, p2));
//    }

//    if (dis_type == Euclidean)
//    {
//        DistanceEuclidean<Point> op;
//        return (op(p1, p2));
//    }

//    if (dis_type == SquaredEuclidean)
//    {
//        DistanceSquaredEuclidean<Point> op;
//        return (op(p1, p2));
//    }
//}

//! @}

/*! @name Distances window to window.
 * \brief From a window to a window.
 */
//! @{
//! Distances window to window from a grid matching to
//! a grid matched
double matchingDistance(NeuralNet<Point2D, GLfloat>* nn, Point2D node_id);




//! KERNEL FUNCTION
//! Kernel call by object
//! Standard Kernel call with objects
template<class Grid, class DisGrid, class O>
KERNEL void K_BinaryOp(Grid g1, Grid g2, DisGrid gr, O op)
{
    KER_SCHED(g1.getWidth(), g1.getHeight())

    if (_x < g1.getWidth() && _y < g1.getHeight()) {
        gr[_y][_x] = op(g1[_y][_x], g2[_y][_x]);
    }

    END_KER_SCHED

    SYNCTHREADS;
}

//// Hongjian's Debug
////! Test program
//template <class Node,
//          typename DisNode,
//          size_t SXX,
//          size_t SYY,
//          class O>
//class TestDistancesMatching {

//    Node initNode1;
//    Node initNode2;
//    DisNode initNode3;

//public:

//    TestDistancesMatching(Node n1, Node n2, DisNode n3) :
//        initNode1(n1),
//        initNode2(n2),
//        initNode3(n3) {}

//    void run() {
//        int devID = 0;
//        cudaError_t error;
//        cudaDeviceProp deviceProp;
//        error = cudaGetDevice(&devID);

//        if (error != cudaSuccess)
//        {
//            printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
//        }

//        error = cudaGetDeviceProperties(&deviceProp, devID);

//        if (deviceProp.computeMode == cudaComputeModeProhibited)
//        {
//            fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
//            exit(EXIT_SUCCESS);
//        }

//        if (error != cudaSuccess)
//        {
//            printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
//        }
//        else
//        {
//            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
//        }

//        cout << "debut test GPU Binary Op on GLfloat ..." << endl;
//        const size_t SX = SXX, SY = SYY;
//        // Creation de grille en local
//        Grid<Node> gdf1(SX, SY), gdf2(SX, SY);
//        Grid<DisNode> gdf(SX, SY);
//        gdf1 = Node(initNode1);
//        gdf2 = Node(initNode2);
//        gdf  = DisNode(initNode3);

//        cout << "Creation de grilles sur device GPU ..." << endl;
//        // Creation de grilles sur device GPU
//#if TEST_CODE
//        Grid<Node> gpu_gdf(SX, SY), gpu_gdf1(SX, SY), gpu_gdf2(SX, SY);
//        // clean local memory
//        gpu_gdf.freeMem();
//        gpu_gdf1.freeMem();
//        gpu_gdf2.freeMem();

//#else
//        Grid<Node> gpu_gdf1, gpu_gdf2;
//        Grid<DisNode> gpu_gdf;
//#endif
//        gpu_gdf.gpuResize(SX, SY);
//        gpu_gdf1.gpuResize(SX, SY);
//        gpu_gdf2.gpuResize(SX, SY);

//        // cuda timer
//        cudaEvent_t start,stop;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, 0);

//        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
//        double x0;
//        // Calcul duree d'execution en ms via clock() / CLOCKS_PER_SEC
//        double xf;
//        x0 = clock();

//        // Affichage
//        (ofstream&) std::cout << "gdf1 = " << endl;
//        (ofstream&) std::cout << gdf1 << endl;
//        (ofstream&) std::cout << "gdf2 = " << endl;
//        (ofstream&) std::cout << gdf2 << endl;

//        cout << gdf.getWidth() << endl;
//        cout << gdf.getHeight() << endl;
//        cout << gdf.getPitch() << endl;

//        cout << gpu_gdf.getWidth() << endl;
//        cout << gpu_gdf.getHeight() << endl;
//        cout << gpu_gdf.getPitch() << endl;

//        cout << "Appel du Kernel ..." << endl;
//        for (int i = 0; i < 1; ++i) {

//            // Copie des grilles CPU -> GPU
//            gdf1.gpuCopyHostToDevice(gpu_gdf1);
//            gdf2.gpuCopyHostToDevice(gpu_gdf2);
//            gdf.gpuCopyHostToDevice(gpu_gdf);

////#if TEST_CODE
//            // Kernel call with class parameters
//            KER_CALL_THREAD_BLOCK(b, t,
//                                  4, 4,
//                                  gpu_gdf1.getWidth(),
//                                  gpu_gdf1.getHeight());
//            K_BinaryOp _KER_CALL_(b, t) (
//                        gpu_gdf1,
//                        gpu_gdf2,
//                        gpu_gdf,
//                        O());
////#endif

//#if TEST_CODE
//            Node* d_g1 = gpu_gdf1.getData();
//            Node* d_g2 = gpu_gdf2.getData();
//            Node* d_gr = gpu_gdf.getData();
//            KER_CALL_THREAD_BLOCK(b2, t2,
//                                  4, 4,
//                                  gpu_gdf1.getWidth(),
//                                  gpu_gdf1.getHeight());
//            K_BinaryOp<Node> _KER_CALL_(b2, t2) (
//                        d_g1,
//                        d_g2,
//                        d_gr,
//                        gpu_gdf1.getWidth(),
//                        gpu_gdf1.getHeight(),
//                        gpu_gdf1.getStride(),
//                        O());
//#endif
//            // Copie du resultat GPU -> CPU
//            gdf.gpuCopyDeviceToHost(gpu_gdf);
////            gdf1 = gdf;
//        }//for

//        cout << "Affichage du resultat a la console ..." << endl;
//        // Affichage du resultat à la console
//        (ofstream&) std::cout << "gdf = " << endl;
//        (ofstream&) cout << gdf << endl;

//        // cpu timer
//        cout << "CPU Time : " << (clock() - x0)/CLOCKS_PER_SEC << endl;

//        // cuda timer
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        GLfloat elapsedTime;
//        cudaEventElapsedTime(&elapsedTime, start, stop); //Resolution ~0.5ms
//        cout << "GPU Execution Time: " <<  elapsedTime << " ms" << endl;
//        cout << endl;

//        // Explicit
//        gpu_gdf.gpuFreeMem();
//        gpu_gdf1.gpuFreeMem();
//        gpu_gdf2.gpuFreeMem();
//        gdf.freeMem();
//        gdf1.freeMem();
//        gdf2.freeMem();
//    }
//};

//! @}

//! @}
}//namespace components

#endif // DISTANCES_H
