#include <iostream>
#include <fstream>

#include "converter.h"

using namespace std;

#define INPUT_SOURCE_IMAGE  "./data/tsukubaL.png" // the source left image to extract colors
#define INPUT_DISPARITY_IMAGE  "./data/tsukuba_groundtruth.png" // the disparity map
#define INPUT_GRID_2D_POINTS  "./data/result.grid2dpts" // the SOM grid

#define OUTPUT_DISPARITY_DATA  "./data/tsukuba_groundtruth_2.data" // the disparity map in text format
#define OUTPUT_MAT_3D_POINTS  "./data/tsukuba_groundtruth_2.mat3dpts" // the disparity converted to 3D points
#define OUTPUT_MAT_OF_COLORS  "./data/tsukuba_groundtruth_2.matcolors" // the colors in text format
#define OUTPUT_GRID_3D_POINTS  "./data/result.grid3dpts" // the SOM grid converted to 3D points
#define OUTPUT_GRID_OF_COLORS  "./data/result.gridcolors" // the SOM grid converted to 3D points

#define SCALE_FACTOR    16.0      // depend of the disparity map
#define BASE_LINE       0.16   // in meters, 3D coordinates are also in meters
#define FOCAL_DISTANCE  374.0   // in pixels (3740 indicated in middlebury database)
#define BACKGROUND_DISPARITY 80.0/SCALE_FACTOR // disparity 0 is replaced by BACKGROUND_DISPARITY

Converter::Converter()
{
    // Input
    source_image = new QImage();
    disparity_image = new QImage();
    grid_2D_points = new Grid2DPoints();

    // Output
    disparity_data = new TMatPix();
    disparity_data_for_meshing = new TMatPix();
    mat_3D_points = new Mat3DPoints();
    mat_of_colors = new Mat3DPoints();
    grid_3D_points = new Grid3DPoints();
    grid_of_colors = new Grid3DPoints();

//    doConversions();
}

Converter::~Converter()
{
    // Input
    delete source_image;
    delete disparity_image;
    delete grid_2D_points;

    // Output
    delete disparity_data;
    delete disparity_data_for_meshing;
    delete mat_3D_points;
    delete mat_of_colors;
    delete grid_3D_points;
    delete grid_of_colors;

}

void Converter::initialize(ConfigParam* cp) {
    this->param = cp;
}

void Converter::doConversions() {
    double disparity;

//    ifstream fi;
//    fi.open(param->fileDisparityDataFromGroundTruth.c_str());
//    if (!fi) {
//        std::cout << "... do conversions ..." << endl;
//    }
//    else {
//        std::cout << "... conversions done ..." << endl;
//        fi.close();
//        return;
//    }

    // Read the input files
    readData();
    cout << endl << "Reading performed" << endl;

    int width = source_image->width();
    int height = source_image->height();

    // Make the conversions

    // Conversion to matrix of colors
    mat_of_colors->resize(width, height);

    for( int _y = 0; _y < height; _y++ )
    {
        for ( int _x = 0; _x < width; _x++ )
        {
            mat_of_colors->data[_y][_x].x = qRed(source_image->pixel(_x,_y));
            mat_of_colors->data[_y][_x].y = qGreen(source_image->pixel(_x,_y));
            mat_of_colors->data[_y][_x].z = qBlue(source_image->pixel(_x,_y));
        }
    }

    // Conversion of disparity values
    int _dispValue = 0;
    disparity_data->resize(width, height);
    for( int _y = 0; _y < height; _y++ )
    {
        for ( int _x = 0; _x < width; _x++ )
        {
            QRgb gray = disparity_image->pixel( _x, _y );
            _dispValue = qGray( gray );
            disparity_data->setValue( _x, _y, _dispValue);
        }
    }

    // Filtrage and contrast for meshing
    int _disp;
    disparity_data_for_meshing->resize(width, height);
    for( int _y = 0; _y < height; _y++ )
    {
        for ( int _x = 0; _x < width; _x++ )
        {
            _disp = disparity_data->getValue( _x, _y);
            if (_disp <= (int) param->minMeshDisparity)
                _disp = 0;
            else
                _disp = _disp * _disp * _disp;//* disp
            disparity_data_for_meshing->setValue( _x, _y, _disp);
        }
    }

    // Conversion to 3D points
    mat_3D_points->resize(width, height);
    for ( int _y = 0; _y < height; _y++ )
    {
        for ( int _x = 0; _x < width; _x++ )
        {
            if (disparity_data->getValue(_x, _y) == 0)
            {
                disparity = param->backgroundDisparity / param->scaleFactor;//BACKGROUND_DISPARITY;
            }
            else
            {
                disparity = disparity_data->getValue(_x, _y) / param->scaleFactor;//SCALE_FACTOR;
            }
//            mat_3D_points->data[_y][_x].z = -FOCAL_DISTANCE * BASE_LINE / disparity;
//            mat_3D_points->data[_y][_x].x = (double) (_x - width/2) * BASE_LINE / disparity;
//            mat_3D_points->data[_y][_x].y = (double) -(_y - height/2) * BASE_LINE / disparity;
            mat_3D_points->data[_y][_x].z = -param->focalDistance * param->baseLine / disparity;
            mat_3D_points->data[_y][_x].x = (double) (_x - width/2) * param->baseLine / disparity;
            mat_3D_points->data[_y][_x].y = (double) -(_y - height/2) * param->baseLine / disparity;
        }
    }

    // Conversion of 2D grid to 3D grid
    double _x, _y;
    grid_3D_points->resize(grid_2D_points->width, grid_2D_points->height);
    for ( int _h = 0; _h < grid_2D_points->height; _h++ )
    {
        for ( int _w = 0; _w < grid_2D_points->width; _w++ )
        {
            _x = grid_2D_points->data[_h][_w].x;
            _y = grid_2D_points->data[_h][_w].y;

            int __x = (int) _x;
            if (_x >= __x + 0.5)
                __x = __x + 1;
            int __y = (int) _y;
            if (_y >= __y + 0.5)
                __y = __y + 1;
//            grid_3D_points->data[_h][_w].z = mat_3D_points->data[__y][__x].z;
//            grid_3D_points->data[_h][_w].x = mat_3D_points->data[__y][__x].x;
//            grid_3D_points->data[_h][_w].y = mat_3D_points->data[__y][__x].y;

            if (disparity_data->getValue(__x, __y) == 0 )
            {
                disparity = param->backgroundDisparity/param->scaleFactor;//BACKGROUND_DISPARITY;
            }
            else
            {
                disparity = disparity_data->getValue(__x, __y) / param->scaleFactor;//SCALE_FACTOR ;
            }
//            grid_3D_points->data[_h][_w].z = -FOCAL_DISTANCE * BASE_LINE / disparity;
//            grid_3D_points->data[_h][_w].x = (_x - width/2) * BASE_LINE / disparity;
//            grid_3D_points->data[_h][_w].y = -(_y - height/2) * BASE_LINE / disparity;
            grid_3D_points->data[_h][_w].z = -param->focalDistance * param->baseLine / disparity;
            grid_3D_points->data[_h][_w].x = (_x - width/2) * param->baseLine / disparity;
            grid_3D_points->data[_h][_w].y = -(_y - height/2) * param->baseLine / disparity;
        }
    }

    // Conversion of 2D grid to color grid
    grid_of_colors->resize(grid_2D_points->width, grid_2D_points->height);
    for ( int _h = 0; _h < grid_2D_points->height; _h++ )
    {
        for ( int _w = 0; _w < grid_2D_points->width; _w++ )
        {
            _x = grid_2D_points->data[_h][_w].x;
            _y = grid_2D_points->data[_h][_w].y;

            int __x = (int) _x;
            if (_x >= __x + 0.5)
                __x = __x + 1;
            int __y = (int) _y;
            if (_y >= __y + 0.5)
                __y = __y + 1;
            grid_of_colors->data[_h][_w].x = mat_of_colors->data[__y][__x].x;
            grid_of_colors->data[_h][_w].y = mat_of_colors->data[__y][__x].y;
            grid_of_colors->data[_h][_w].z = mat_of_colors->data[__y][__x].z;
        }
    }

    // Save the ouput files
    writeData();
    cout << endl << "Writing performed" << endl;
}

void Converter::readData() {

    // Load the left source image
//    source_image->load(INPUT_SOURCE_IMAGE);
//    source_image->load(param->inputDisparityImageGroundTruth.c_str());
    source_image->load(param->inputSourceImageLeft.c_str());

    // Load the disparity image
//    disparity_image->load(INPUT_DISPARITY_IMAGE);
    disparity_image->load(param->inputDisparityImageGroundTruth.c_str());

    // Load the grid of 2 points
    ifstream fi;
//    fi.open(INPUT_GRID_2D_POINTS);
    fi.open(param->fileGrid2DPoints.c_str());
    if (!fi) {
        std::cout << "erreur ouverture fileGrid2DPoints" << endl;
    }
    else
        fi >> *grid_2D_points;
    fi.close();

}

void Converter::writeData() {

    // Save the disparity data
    ofstream fo;
//    fo.open(OUTPUT_DISPARITY_DATA);
    fo.open(param->fileDisparityDataFromGroundTruth.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileDisparityDataFromGroundTruth" << endl;
        return;
    }
    fo << *disparity_data;
    fo.close();

    // Save the disparity data for meshing
    fo.open(param->fileDisparityDataForMeshing.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileDisparityDataForMeshing" << endl;
        return;
    }
    fo << *disparity_data_for_meshing;
    fo.close();

    // Save the matrix of 3 points
//    fo.open(OUTPUT_MAT_3D_POINTS);
    fo.open(param->fileMat3DPoints.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileMat3DPoints" << endl;
        return;
    }
    fo << *mat_3D_points;
    fo.close();

    // Save the matrix of colors
//    fo.open(OUTPUT_MAT_OF_COLORS);
    fo.open(param->fileMatOfColors.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileMatOfColors" << endl;
        return;
    }
    fo << *mat_of_colors;
    fo.close();

    // Save the grid of 3D points
//    fo.open(OUTPUT_GRID_3D_POINTS);
    fo.open(param->fileGrid3DPoints.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileGrid3DPoints" << endl;
        return;
    }
    fo << *grid_3D_points;
    fo.close();

    // Save the grid of colors
//    fo.open(OUTPUT_GRID_OF_COLORS);
    fo.open(param->fileGridOfColors.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileGridOfColors" << endl;
        return;
    }
    fo << *grid_of_colors;
    fo.close();
}

void Converter::updateGrid3D() {
    double disparity = param->backgroundDisparity/param->scaleFactor;//BACKGROUND_DISPARITY;

    // Load the grid of 2 points
    ifstream fi;
//    fi.open(INPUT_GRID_2D_POINTS);
    fi.open(param->fileGrid2DPoints.c_str());
    if (!fi) {
        std::cout << "erreur ouverture fileGrid2DPoints" << endl;
    }
    else
        fi >> *grid_2D_points;
    fi.close();
    cout << endl << "Reading performed" << endl;

    int width = source_image->width();
    int height = source_image->height();

    // Make the conversions

    // Conversion of 2D grid to 3D grid
    double _x, _y;
    grid_3D_points->resize(grid_2D_points->width, grid_2D_points->height);
    for ( int _h = 0; _h < grid_2D_points->height; _h++ )
    {
        for ( int _w = 0; _w < grid_2D_points->width; _w++ )
        {
            _x = grid_2D_points->data[_h][_w].x;
            _y = grid_2D_points->data[_h][_w].y;

            int __x = (int) _x;
            if (_x >= __x + 0.5)
                __x = __x + 1;
            int __y = (int) _y;
            if (_y >= __y + 0.5)
                __y = __y + 1;

            if (disparity_data->getValue(__x, __y) == 0 )
            {
                disparity = param->backgroundDisparity/param->scaleFactor;//BACKGROUND_DISPARITY;
            }
            else
            {
                disparity = disparity_data->getValue(__x, __y) / param->scaleFactor;//SCALE_FACTOR ;
            }
//            grid_3D_points->data[_h][_w].z = -FOCAL_DISTANCE * BASE_LINE / disparity;
//            grid_3D_points->data[_h][_w].x = (_x - width/2) * BASE_LINE / disparity;
//            grid_3D_points->data[_h][_w].y = -(_y - height/2) * BASE_LINE / disparity;
            grid_3D_points->data[_h][_w].z = -param->focalDistance * param->baseLine / disparity;
            grid_3D_points->data[_h][_w].x = (_x - width/2) * param->baseLine / disparity;
            grid_3D_points->data[_h][_w].y = -(_y - height/2) * param->baseLine / disparity;
        }
    }

    // Conversion of 2D grid to color grid
    grid_of_colors->resize(grid_2D_points->width, grid_2D_points->height);
    for ( int _h = 0; _h < grid_2D_points->height; _h++ )
    {
        for ( int _w = 0; _w < grid_2D_points->width; _w++ )
        {
            _x = grid_2D_points->data[_h][_w].x;
            _y = grid_2D_points->data[_h][_w].y;

            int __x = (int) _x;
            if (_x >= __x + 0.5)
                __x = __x + 1;
            int __y = (int) _y;
            if (_y >= __y + 0.5)
                __y = __y + 1;
            grid_of_colors->data[_h][_w].x = mat_of_colors->data[__y][__x].x;
            grid_of_colors->data[_h][_w].y = mat_of_colors->data[__y][__x].y;
            grid_of_colors->data[_h][_w].z = mat_of_colors->data[__y][__x].z;
        }
    }

    ofstream fo;
    // Save the grid of 3D points
//    fo.open(OUTPUT_GRID_3D_POINTS);
    fo.open(param->fileGrid3DPoints.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileGrid3DPoints" << endl;
        return;
    }
    fo << *grid_3D_points;
    fo.close();

    // Save the grid of colors
//    fo.open(OUTPUT_GRID_OF_COLORS);
    fo.open(param->fileGridOfColors.c_str());
    if (!fo) {
        std::cout << "erreur ouverture fileGridOfColors" << endl;
        return;
    }
    fo << *grid_of_colors;
    fo.close();

    cout << endl << "Writing performed" << endl;
}

