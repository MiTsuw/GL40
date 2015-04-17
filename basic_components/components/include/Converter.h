#ifndef CONVERTER_H
#define CONVERTER_H
/*
 ***************************************************************************
 *
 * Author : W. Qiao, J.C. Creput
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>

#include "macros_cuda.h"
#include "GridOfNodes.h"
#include "NeuralNet.h"
#include "Node.h"
#include "ImageRW.h"

using namespace std;
using namespace components;

#define SCALE_FACTOR    4.0      // depend of the disparity map
#define BASE_LINE       0.16   // in meters, 3D coordinates are also in meters
#define FOCAL_DISTANCE  374.0   // in pixels (3740 indicated in middlebury database)
#define BACKGROUND_DISPARITY 80.0/SCALE_FACTOR // disparity 0 is replaced by BACKGROUND_DISPARITY
#define MIN_MESH_DISPARITY 50
#define USERDISPARITYMAP 0

//Venus
//scaleFactor = 8.0
//baseLine = 0.16
//focalDistance = 374.0
//#focalDistance = 935.0
//#focalDistance = 3740.0
//disparityRange = 20
//backgroundDisparity = 25
//minMeshDisparity = 25

//Teddy
//scaleFactor = 4.0
//baseLine = 0.16
//focalDistance = 374.0
//#focalDistance = 935.0
//#focalDistance = 3740.0
//disparityRange = 60
//backgroundDisparity = 68
//minMeshDisparity = 68

//Cones
//scaleFactor = 4.0
//baseLine = 0.16
//focalDistance = 374.0
//#focalDistance = 935.0
//#focalDistance = 3740.0
//disparityRange = 60
//backgroundDisparity = 68.0
//minMeshDisparity = 90

//Tsukuba
//scaleFactor = 16.0
//baseLine = 0.16
//focalDistance = 374.0
//disparityRange = 60
//backgroundDisparity = 80.0
//minMeshDisparity = 80

namespace components
{

class Converter
{
public:

    void doConversions(NN& nnI, NN& nn, NNP3D& nnIo, NNP3D& nno){

        int width = nnI.colorMap.width;
        int height = nnI.colorMap.height;

        //! Conversion of nnI.colorMap to nnIo.adaptiveMap that has the 3D Euclidean coordinates
        double disparity=0;
        nnIo.adaptiveMap.resize(width, height);
        for (int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                if (nnI.densityMap.get(_x, _y) == 0)
                {
                    disparity = BACKGROUND_DISPARITY;
                }
                else
                {
                    disparity = nnI.densityMap.get(_x, _y)/SCALE_FACTOR;
                }
                nnIo.adaptiveMap[_y][_x][2] = -FOCAL_DISTANCE * BASE_LINE / disparity;
                nnIo.adaptiveMap[_y][_x][0] = (double) (_x - width/2) * BASE_LINE / disparity;
                nnIo.adaptiveMap[_y][_x][1] = (double) -(_y - height/2) * BASE_LINE / disparity;
            }
        }

        //! Conversion of nn.adaptiveMap(2d) to nno.adaptiveMap(3d) that has the 3D Euclidean coordinates
        int gridWidth = nn.adaptiveMap.width;
        int gridHeight = nn.adaptiveMap.height;
        double _x, _y;
        disparity = 0;
        nno.adaptiveMap.resize(gridWidth, gridHeight);

        for (int _h = 0; _h < gridHeight; _h++)
        {
            for (int _w = 0; _w < gridWidth; _w++)
            {
                _x = nn.adaptiveMap[_h][_w][0];
                _y = nn.adaptiveMap[_h][_w][1];

                //! JCC
                //Point2D pp(_x, _y);
                //PointCoord ppp = vgd.FRound(pp);

                int __x = (int) _x;
                if (_x >= __x + 0.5)
                    __x = __x + 1;
                int __y = (int) _y;
                if (_y >= __y + 0.5)
                    __y = __y + 1;

                //! JCC 130315 : modif
                if (__x < 0)
                    __x = 0;
                if (__x >= nnI.densityMap.width)
                    __x = nnI.densityMap.width - 1;
                if (__y < 0)
                    __y = 0;
                if (__y >= nnI.densityMap.height)
                    __y = nnI.densityMap.height - 1;

                //! JCC 130315 : modif
                if (!USERDISPARITYMAP) {
                    disparity = nnI.densityMap.get(__x, __y) /SCALE_FACTOR;
                }
                else {
                    if (nnI.densityMap.get(__x, __y) == 0 )
                    {
                        disparity = BACKGROUND_DISPARITY;
                    }
                    else
                    {
                        disparity = nnI.densityMap.get(__x, __y) /SCALE_FACTOR;
                    }
                }
                nno.adaptiveMap[_h][_w][2] = -FOCAL_DISTANCE * BASE_LINE / disparity;
                nno.adaptiveMap[_h][_w][0] = (_x - width/2) * BASE_LINE / disparity;
                nno.adaptiveMap[_h][_w][1] = -(_y - height/2) * BASE_LINE / disparity;
            }
        }

        //! Conversion of nn.adaptiveMap to nno.colorMap that has the color RGB the same as nnI.colorMap
        nno.colorMap.resize(gridWidth, gridHeight);
        for (int _h = 0; _h < gridHeight; _h++)
        {
            for (int _w = 0; _w < gridWidth; _w++)
            {
                _x = nn.adaptiveMap[_h][_w][0];
                _y = nn.adaptiveMap[_h][_w][1];

                int __x = (int) _x;
                if (_x >= __x + 0.5)
                    __x = __x + 1;
                int __y = (int) _y;
                if (_y >= __y + 0.5)
                    __y = __y + 1;

                //! JCC 130315 : modif
                bool debord = false;
                if (__x < 0) {
                    debord = true;
                    __x = 0;
                }
                if (__x >= nnI.colorMap.getWidth()) {
                    debord = true;
                    __x = nnI.colorMap.getWidth() - 1;
                }
                if (__y < 0) {
                    debord = true;
                    __y = 0;
                }
                if (__y >= nnI.colorMap.getHeight()) {
                    debord = true;
                    __y = nnI.colorMap.getHeight() - 1;
                }

                //! JCC 130315 : modif
                if (debord) {
                    nno.colorMap[_h][_w][0] = 255;
                    nno.colorMap[_h][_w][1] = 255;
                    nno.colorMap[_h][_w][2] = 255;
                }
                else {
                    nno.colorMap[_h][_w][0] = nnI.colorMap[__y][__x][0];
                    nno.colorMap[_h][_w][1] = nnI.colorMap[__y][__x][1];
                    nno.colorMap[_h][_w][2] = nnI.colorMap[__y][__x][2];
                }
            }
        }
    }

}; // Converter

} // namespace components

#endif // CONVERTER.H
