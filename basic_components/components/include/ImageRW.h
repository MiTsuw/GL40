 #ifndef IMAGERW_H
#define IMAGERW_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, W. Qiao
 * Creation date : Mar. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>

#include <QtGui/QImage>

#include "macros_cuda.h"
#include "GridOfNodes.h"
#include "Node.h"
#include "NeuralNet.h"

#define IRW_COMPLETE 0

#define SFX_ADAPTIVE_MAP_IMAGE  "_adaptiveMap.pgm"
#define SFX_COLOR_MAP_IMAGE  ".png"
#define SFX_DENSITY_MAP_IMAGE  "_groundtruth.pgm"

using namespace std;
using namespace components;

namespace components
{

template <class Point,
          class Value>
class ImageRW
{
public:

    int getPos(string str){

        std::size_t pos = str.find('.',  0);
        if (pos == std::string::npos)
            pos = str.length();

        //! no-matter aaa_xx.xx or aaa.xx or aaa_xx,  aaa.xx_xx
        //! we can always get the first three aaa and the same pos =  = 3
        std::size_t posTiret = str.find('_',  0);

        if(pos > posTiret)
            pos = posTiret; //! ensure to get the aaa in any name format aaa_xxx or aaa_xxx.lll

        return pos;
    }

    void read(string str, NeuralNet<Point, Value>& nn) {

        //! Input
        QImage colorMapImage; //! input_color_image
        QImage densityMapImage; //! input_densityMapImage

        int pos = getPos(str);
        string str_sub;

#if IRW_COMPLETE
        QImage adaptiveMapImage; //! input_adaptiveMap

        //! load the adaptiveMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_ADAPTIVE_MAP_IMAGE);
        adaptiveMapImage.load(str_sub.c_str());
        cout << "base_name_adaptive= " << str_sub << endl;

#endif

        //! load colorMap
        str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE);
        colorMapImage.load(str_sub.c_str());

        //! Load densityMap(groundtruth image)
        str_sub = str.substr(0,  pos);
        str_sub.append(SFX_DENSITY_MAP_IMAGE);
        densityMapImage.load(str_sub.c_str());

        int width = colorMapImage.width(); //! width of input_color_image
        int height = colorMapImage.height(); //! height of input_color_image

        //! fill the colorMap of nn
        nn.colorMap.resize(width,  height);
        for(int _y = 0; _y < height; _y++)
        {
            for (int _x = 0; _x < width; _x++)
            {
                //! fill colorMap
                nn.colorMap[_y][_x][0] = qRed(colorMapImage.pixel(_x, _y));
                nn.colorMap[_y][_x][1] = qGreen(colorMapImage.pixel(_x, _y));
                nn.colorMap[_y][_x][2] = qBlue(colorMapImage.pixel(_x, _y));
            }
        }

        //! fill the density_map of nn from the groundtruth.image
        int _dispValue = 0;
        nn.densityMap.resize(width, height);
        for( int _y = 0; _y < height; _y++ )
        {
            for ( int _x = 0; _x < width; _x++ )
            {
                QRgb gray = densityMapImage.pixel( _x,  _y );
                _dispValue = qGray( gray );
                nn.densityMap.set( _x, _y, _dispValue);
            }
        }
    }

    //! write the NN_colorMap to image
    void write(string str, NeuralNet<Point, Value>& nn) {

        //! Convert grid to qimages
        int width = nn.colorMap.width;
        int height = nn.colorMap.height;
        QImage reImage = QImage(width, height, QImage::QImage::Format_RGB32);

        for (int _y = 0; _y < height;_y++)
        {
            for (int _x = 0; _x < width;_x++)
            {
                int r = nn.colorMap[_y][_x][0];
                int g = nn.colorMap[_y][_x][1];
                int b = nn.colorMap[_y][_x][2];
                QRgb pixel_color = qRgb(r, g, b);

                reImage. setPixel(_x, _y, pixel_color);
            }
        }

        int pos = getPos(str);
        string str_sub = str.substr(0, pos);
        str_sub.append(SFX_COLOR_MAP_IMAGE);
        reImage.save(str_sub.c_str());
    }
};

typedef ImageRW<Point2D, GLfloat> IRW;

} // namespace components

#endif // IMAGERW_H
