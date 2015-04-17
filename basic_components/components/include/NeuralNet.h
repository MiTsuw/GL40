#ifndef NEURALNET_H
#define NEURALNET_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang, W. Qiao
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#ifdef CUDA_CODE
#include <curand_kernel.h>
#include <sm_20_atomic_functions.h>

#endif
#include <time.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
//#include <boost/geometry/geometries/point.hpp>

#include "macros_cuda.h"
#include "Node.h"
#include "Objectives.h"
#include "GridOfNodes.h"

#define NN_COMPLETE_RW 0 // 0 is false, != 0 is true

using namespace std;

#define NN_SFX_ADAPTIVE_MAP_IMAGE  ".points"
#define NN_SFX_COLOR_MAP_IMAGE  ".gridcolors"
#define NN_SFX_DENSITY_MAP_IMAGE  ".data"

namespace components
{

/*!
 * \brief The NeuralNet struct
 * JCC 06/03/15 : A neural networks is a set of grids
 * that represent its different attributes
 * (position, color, active, fixed ...) where
 * we must at least find :
 * - a grid of 2D/3D Points that represent locations in euclidean plane
 * - a grid of density value.
 *
 * Example :
 * - Adaptive mesh : it is a NeuralNet<Point2D, GLfloat>
 * that moves onto 2D/3D space, with a density value
 * associated to each node.
 * - Density mesh : it is a NeuralNet<Point2D, GLfloat>
 * that represents a density distribution, where each
 * a density value is associated to each node in the plane.
 * It is generally represented as an image pixel map.
 * - Ring Mesh : it is a NeuralNet<Point2D, GLfloat>
 * organized as a ring (w=2*N) or a set of ring.
 * - City mesh : it is a set of cities implemented
 * as a grid, where each city has a location in the plane,
 * and a density value associated to it.
 *
 * Then, the Goal of Adaptive Meshing or Grid Matching Problem is to
 * realize "the" best matching of a Neural Network (or mesh) onto another
 * Neural Network (or mesh).
 *
 * Examples :
 * - Mesh onto Density Mesh (meshing pb)
 * - Ring onto City Mesh (TSP pb)
 * - Clustering k-mean : Mesh at given level to a low level mesh
 * - Image Left Mesh onto Image Right Mesh (stereo)
 * - Image First Mesh onto Second Image Mesh (flow2D)
 * - Match between four Image Meshes (flow3D)
 */
template <typename Point,
          typename Value>
struct NeuralNet {

    NeuralNet() {}

    //! Differential objectives
    //! that evaluate adaptation or matching
    //! k-mean distortion, length, cost, smoothing, gdtruth ...
    Grid<AMObjectives> objectivesMap;

    //! \brief High level cluster centers grid.
    //! Can be a mesh or a ring,
    //! or any grid of nodes that can move into
    //! a space of any dimension (1,2,3 usual)
    Grid<Point> adaptiveMap;

    //! Pattern of activation/fixation
    Grid<bool> activeMap;
    Grid<bool> fixedMap;

    //! Colored grid and gray values
    Grid<Point3D> colorMap;
    Grid<GLint> grayValueMap;

    //! \brief Low level density map for point extraction.
    //! Can be a density distribution or a grid of cities,
    //! or any grid of nodes that return a scalar intensity value
    //! or at least have a relation order <= and >= comparators
    //! for roulette wheel extraction,
    //! and relative addition + operator.
    Grid<Value> densityMap;//level 1 density map

public:

    DEVICE_HOST explicit NeuralNet(int w, int h) :
        objectivesMap(w, h),
        adaptiveMap(w, h),
        activeMap(w, h),
        fixedMap(w, h),
        colorMap(w, h),
        grayValueMap(w, h),
        densityMap(w, h) { }

    /*! @name Globales functions specific for controling the GPU.
     * \brief Memory allocation and communication. Useful
     * for mixte utilisation.
     * @{
     */

    //! For CPU side
    void allocMem() {
        objectivesMap.allocMem();
        adaptiveMap.allocMem();
        activeMap.allocMem();
        fixedMap.allocMem();
        colorMap.allocMem();
        grayValueMap.allocMem();
        densityMap.allocMem();
    }

    void freeMem() {
        objectivesMap.freeMem();
        adaptiveMap.freeMem();
        activeMap.freeMem();
        fixedMap.freeMem();
        colorMap.freeMem();
        grayValueMap.freeMem();
        densityMap.freeMem();
    }

    void resize(int w, int h) {
        objectivesMap.resize(w, h);
        adaptiveMap.resize(w, h);
        activeMap.resize(w, h);
        fixedMap.resize(w, h);
        colorMap.resize(w, h);
        grayValueMap.resize(w, h);
        densityMap.resize(w, h);
    }

    //! For GPU side
    void gpuAllocMem() {
        objectivesMap.gpuAllocMem();
        adaptiveMap.gpuAllocMem();
        activeMap.gpuAllocMem();
        fixedMap.gpuAllocMem();
        colorMap.gpuAllocMem();
        grayValueMap.gpuAllocMem();
        densityMap.gpuAllocMem();
    }

    void gpuFreeMem() {
        objectivesMap.gpuFreeMem();
        adaptiveMap.gpuFreeMem();
        activeMap.gpuFreeMem();
        fixedMap.gpuFreeMem();
        colorMap.gpuFreeMem();
        grayValueMap.gpuFreeMem();
        densityMap.gpuFreeMem();
    }

    void gpuResize(int w, int h) {
        objectivesMap.gpuResize(w, h);
        adaptiveMap.gpuResize(w, h);
        activeMap.gpuResize(w, h);
        fixedMap.gpuResize(w, h);
        colorMap.gpuResize(w, h);
        grayValueMap.gpuResize(w, h);
        densityMap.gpuResize(w, h);
    }

    //! HOST to DEVICE
    void gpuCopyHostToDevice(NeuralNet & gpuNeuralNet) {
        objectivesMap.gpuCopyHostToDevice(gpuNeuralNet.objectivesMap);
        adaptiveMap.gpuCopyHostToDevice(gpuNeuralNet.adaptiveMap);
        activeMap.gpuCopyHostToDevice(gpuNeuralNet.activeMap);
        fixedMap.gpuCopyHostToDevice(gpuNeuralNet.fixedMap);
        colorMap.gpuCopyHostToDevice(gpuNeuralNet.colorMap);
        grayValueMap.gpuCopyHostToDevice(gpuNeuralNet.grayValueMap);
        densityMap.gpuCopyHostToDevice(gpuNeuralNet.densityMap);
    }

    //! DEVICE TO HOST
    void gpuCopyDeviceToHost(NeuralNet & gpuNeuralNet) {
        objectivesMap.gpuCopyDeviceToHost(gpuNeuralNet.objectivesMap);
        adaptiveMap.gpuCopyDeviceToHost(gpuNeuralNet.adaptiveMap);
        activeMap.gpuCopyDeviceToHost(gpuNeuralNet.activeMap);
        fixedMap.gpuCopyDeviceToHost(gpuNeuralNet.fixedMap);
        colorMap.gpuCopyDeviceToHost(gpuNeuralNet.colorMap);
        grayValueMap.gpuCopyDeviceToHost(gpuNeuralNet.grayValueMap);
        densityMap.gpuCopyDeviceToHost(gpuNeuralNet.densityMap);
    }
    //! @}

    int getPos(string str){

        std::size_t pos = str.find('.',  0);
        if (pos ==  std::string::npos)
            pos = str.length();

        //! no-matter aaa_xx.xx or aaa.xx or aaa_xx,  aaa.xx_xx
        //! we can always get the first three aaa and the same pos =  = 3
        std::size_t  posTiret = str.find('_',  0);

            if(pos > posTiret)
                pos = posTiret; //! ensure to get the aaa in any name format aaa_xxx or aaa_xxx.lll

        return pos;
    }

    void read(string str) {

        int pos = getPos(str);
        ifstream fi;
        //! read adaptiveMap
        string str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_ADAPTIVE_MAP_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read " << str_sub << endl; }
        fi >> adaptiveMap;
        fi.close();

        //! read colorMap
        str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_COLOR_MAP_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> colorMap;
        fi.close();

        //! read densityMap
        str_sub = str.substr(0, pos);
        str_sub.append(NN_SFX_DENSITY_MAP_IMAGE);
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> densityMap;
        fi.close();

#if NN_COMPLETE_RW
        //! read objectivesMap
        str_sub = str.substr(0, pos);
        str_sub.append(".objectivesMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> objectivesMap;
        fi.close();

        //! read fixedMap
        str_sub = str.substr(0, pos);
        str_sub.append(".fixedMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> fixedMap;
        fi.close();

        //! read grayValueMap;
        str_sub = str.substr(0, pos);
        str_sub.append(".grayValueMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> grayValueMap;
        fi.close();

        //! read activeMap;
        str_sub = str.substr(0, pos);
        str_sub.append(".activeMap");
        fi.open(str_sub.c_str() );
        if (!fi) {
            std::cout << "erreur read: not exits "<< str_sub << endl; }
        fi >> activeMap;
        fi.close();
#endif

    }

    void write(string str) {

        int pos=getPos(str);
        string str_sub;

        ofstream fo;
        //! write adaptiveMap;
        if(adaptiveMap.width != 0 && adaptiveMap.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_ADAPTIVE_MAP_IMAGE);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur  write " << str_sub << endl; }
            fo << adaptiveMap;
            fo.close();
        }

        //! write colorMap
        if(colorMap.width != 0 && colorMap.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_COLOR_MAP_IMAGE);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write " << str_sub << endl; }
            fo << colorMap;
            fo.close();
        }

        //! write densityMap
        if(densityMap.width != 0 && densityMap.height != 0){
            str_sub = str.substr(0, pos);
            str_sub.append(NN_SFX_DENSITY_MAP_IMAGE);
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << densityMap;
            fo.close();
        }

#if NN_COMPLETE_RW
        //! read objectivesMap
        if(objectivesMap.width!=0 || objectivesMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".objectivesMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << objectivesMap;
            fo.close();
        }

        //! read fixedMap
        if(fixedMap.width!=0 || fixedMap.height!=0){
            str_sub = str.substr(0, pos);

            str_sub.append(".fixedMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << fixedMap;
            fo.close();
        }

        //! read activeMap;
        if(activeMap.width!=0 || activeMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".activeMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << activeMap;
            fo.close();
        }

        //! read grayValueMap;
        if(grayValueMap.width!=0 || grayValueMap.height!=0){
            str_sub = str.substr(0, pos);
            str_sub.append(".grayValueMap");
            fo.open(str_sub.c_str() );
            if (!fo) {
                std::cout << "erreur write "<< str_sub << endl; }
            fo << grayValueMap;
            fo.close();
        }

#endif
    }

    //    Grid<Point> getAdaptiveMap() const;
    //    void setAdaptiveMap(const Grid<Point> &value);

    Grid<Point> getAdaptiveMap() const
    {
        return adaptiveMap;
    }

    void setAdaptiveMap(const Grid<Point> &value)
    {
        adaptiveMap = value;
    }

};

typedef NeuralNet<Point2D, GLfloat> NN;
typedef NeuralNet<Point3D, GLfloat> NNP3D;

}//namespace components
#endif // NEURALNET_H
