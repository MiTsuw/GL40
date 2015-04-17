#ifndef GRID_OF_NODES_H
#define GRID_OF_NODES_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
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

#include "macros_cuda.h"
#include "Node.h"
#include "Objectives.h"

#define TEST_CODE 0
#define TEST_PITCH 1 // in that case STRIDE_ALIGNMENT=64 (GPU side)
#define INTERPOLATION 1

#define MIXTE_CPU_GPU_OBJECT 1

using namespace std;

namespace components
{

/*! @name Allocateurs CPU/GPU.
 * \brief Le choix est effectue par template.
 */
//! @{
//! Allocateur local CPU/DEVICE
template <class Node>
struct Allocator2DLocal {

    DEVICE_HOST Node* allocMem(size_t width, size_t height, size_t& pitch) {

        Node* _data;

        pitch = width * sizeof(Node);

        _data = new Node[width * height];

        return _data;
    }

    DEVICE_HOST void freeMem(Node* _data) {

        if (_data != NULL) {
            delete [] _data;
            _data = NULL;
        }
    }
};

//! Allocateur GPU

template <class Node>
struct Allocator2DGPU {

    Node* allocMem(size_t width, size_t height, size_t& pitch) {

        Node* _data;
#ifdef CUDA_CODE
        cudaMallocPitch(
                    (void**)&_data,
                    &pitch,
                    sizeof(Node) * width,
                    height);
#else
        _data = Allocator2DLocal<Node>().allocMem(width, height, pitch);
#endif
        return _data;
    }

    void freeMem(Node* _data) {
#ifdef CUDA_CODE
        if (_data != NULL) {
            cudaFree(_data);
            _data = NULL;
        }
#else
        if (_data)
            delete [] _data;
#endif
    }

};

//! @}

/*!
 * \defgroup Grilles de nodes
 * \brief Espace de nommage components
 * Il comporte les grilles (grids)
 */
/*! @{*/

/*! \brief Classe definissant la structure d'une grille de noeuds (nodes).
 *
 * Une grille est un vecteur de lignes de points (nodes).
 */
//template <class Node,
//          class Alloc = Allocator2DLocal<Node>(),
//          class GPUAlloc = Allocator2DGPU<Node>()>
template <class Node>
struct Grid
{
    int dist_level;
    int dual_level;

    size_t width;
    size_t height;
    size_t pitch; // counted in bytes

    Node* _data;

    typedef Allocator2DLocal<Node> Alloc;
    typedef Allocator2DGPU<Node> GPUAlloc;

    Alloc alloc;
    GPUAlloc gpu_alloc;

public:

    //! Default constructor is private (do not access)
    DEVICE_HOST explicit Grid() : dist_level(0),
        dual_level(0),
        width(0),
        height(0),
        pitch(0),
        _data(NULL) {}

    DEVICE_HOST explicit Grid(int w, int h) : dist_level(0),
        dual_level(0),
        width(w),
        height(h) {
        _data = alloc.allocMem(width, height, pitch);
    }

    DEVICE_HOST explicit Grid(int w, int h, Alloc a, Alloc b) :
        dist_level(0),
        dual_level(0),
        width(w),
        height(h),
        alloc(a),
        gpu_alloc(b) {
        _data = alloc.allocMem(width, height, pitch);
    }

    //! JCC : This code can not exist if using MIXTE CPU/GPU
    //! object, since only the default copy constructor
    //! must be used by CUDA itself
    //! (don't rewrite copy constructor if mixte,
    //! only for host/device specific configuration).
#if MIXTE_CPU_GPU_OBJECT
#else
    //! No cons copy because mixte object CPU/GPU
    //! Only explicit memory alloc/dealloc is allowed
    //! @brief Cons copie
    DEVICE_HOST Grid(Grid& g2) : width(g2.width), height(g2.height) {
        _data = alloc.allocMem(width, height, pitch);
        std::memcpy(_data, g2._data, pitch * height);
    }
    DEVICE_HOST ~Grid() { alloc.freeMem(_data);
    }
#endif

//! JCC 300315 : Suppression of affectation
//! only use default affectation, as cons. copie
//    //! @brief Affectation
//    DEVICE_HOST Grid& operator=(Grid const& g2) {
//#if MIXTE_CPU_GPU_OBJECT
//#else
// JCC : no implicit alloc if MIXTE OBJECT
// but must test if "this" and if different size
//    if (this != &g2 && (width != g2.width
//                        || height != g2.height))
//        this->resize(g2.width, g2.height);
//#endif
//        std::memcpy(_data, g2._data, pitch * height);
//        return *this;
//    }

    //! @brief Affectation of a value (idem reset)
    DEVICE_HOST Grid& operator=(Node const& node) {
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                *(((Node*)((char*)_data + _y * pitch)) + _x) = node;
            }
        }
        return *this;
    }

    DEVICE_HOST Node* getData() { return _data; }

    DEVICE_HOST void resetValue(Node const& node) {
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                *(((Node*)((char*)_data + _y * pitch)) + _x) = node;
            }
        }
    }

    DEVICE_HOST size_t getWidth() const {
        return width;
    }

    DEVICE_HOST size_t getHeight() const {
        return height;
    }

    DEVICE_HOST size_t getPitch() {
        return pitch;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline Node* operator[](std::size_t y) {
        return ((Node*)((char*)_data + y * pitch));
    }

    //! @brief Get coordinate for loop only
    //! mirror out-of-range position
    DEVICE_HOST inline Node const& fetchIntCoor(int x, int y) const {
        if (x < 0) x = abs(x + 1);
        if (y < 0) y = abs(y + 1);
        if (x >= width) x = width * 2 - x - 1;
        if (y >= height) y = height * 2 - y - 1;
        return *((Node*)((char*)_data + y * pitch) + x);
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline Node const& get(std::size_t const x, std::size_t const y) const {
        return *((Node*)((char*)_data + y * pitch) + x);
    }

    //! @brief Set coordinatev for loop only
    DEVICE_HOST inline void set(std::size_t const x, std::size_t const y, Node const& value) {
        *((Node*)((char*)_data + y * pitch) + x) = value;
    }

    //! @brief Auto Addition
    DEVICE_HOST Grid& operator+=(Grid const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) += g2.get(_x, _y);
            }
        }
        return *this;
    }

    //! @brief Auto Difference
    DEVICE_HOST Grid& operator-=(Grid const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) -= g2.get(_x, _y);
            }
        }
        return *this;
    }

    //! @brief Auto Addition
    DEVICE_HOST Grid& operator+=(Node const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) += g2;
            }
        }
        return *this;
    }

    //! @brief Auto Difference
    DEVICE_HOST Grid& operator-=(Node const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) -= g2;
            }
        }
        return *this;
    }

    //! @brief Auto Mult
    DEVICE_HOST Grid& operator*=(Grid const& g2) {
        // op
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*((Node*)((char*)_data + _y * pitch) + _x)) *= g2.get(_x, _y);
            }
        }
        return *this;
    }

#if INTERPOLATION
    //! @brief Get coordinate for loop only
    //! mirror out-of-range position
    //! read from arbitrary position within image using bilinear interpolation
    DEVICE_HOST inline Node const& fetchFloatCoor(GLfloat x, GLfloat y) const {
        // integer parts in floating point format
        GLfloat intPartX, intPartY;
        // get fractional parts of coordinates
        GLfloat dx = fabsf(modff(x, &intPartX));
        GLfloat dy = fabsf(modff(y, &intPartY));
        // assume pixels are squares
        // one of the corners
        int ix0 = (int)intPartX;
        int iy0 = (int)intPartY;
        // mirror out-of-range position
        if (ix0 < 0) ix0 = abs(ix0 + 1);
        if (iy0 < 0) iy0 = abs(iy0 + 1);
        if (ix0 >= width) ix0 = width * 2 - ix0 - 1;
        if (iy0 >= height) iy0 = height * 2 - iy0 - 1;
        // corner which is opposite to (ix0, iy0)
        int ix1 = ix0 + 1;
        int iy1 = iy0 + 1;
        if (ix1 >= width) ix1 = width * 2 - ix1 - 1;
        if (iy1 >= height) iy1 = height * 2 - iy1 - 1;

        GLfloat a = (1.0f - dx) * (1.0f - dy);
        GLfloat b = dx * (1.0f - dy);
        GLfloat c = (1.0f - dx) * dy;
        GLfloat d = dx * dy;
        Node res = (this->get(ix0, iy0) * a);
        res += (this->get(ix1, iy0) * b);
        res += (this->get(ix0, iy1) * c);
        res += (this->get(ix1, iy1) * d);
        return res;
    }
#endif

//! JCC : This code can not exist if using MIXTE CPU/GPU
//! object, since only the default copy constructor
//! must be used by CUDA itself
//! (don't rewrite copy constructor if mixte,
//! only for host/device specific configuration)
#if MIXTE_CPU_GPU_OBJECT
#else
    //Functions returning an object value are not allowed
    //because Grid is mixte object CPU/GPU : in that case
    //for Kernel call global function, the only copy constructor
    //for passing parameters to the device is the default one.
    //Hence, the user copy constructor is suppressed.
    //! @brief Addition
    DEVICE_HOST friend Grid operator+(Grid const& g1, Grid const& g2) {
        Grid g(g1.width, g1.height);

        for (int _y = 0; _y < g1.height; _y++) {
            for (int _x = 0; _x < g1.width; _x++) {
                g.set(_x, _y, g1.get(_x, _y) + g2.get(_x, _y));
            }
        }
        return g;
    }

    //! @brief Difference
    DEVICE_HOST friend Grid operator-(Grid const& g1, Grid const& g2) {
        Grid g(g1.width, g1.height);

        for (int _y = 0; _y < g1.height; ++_y) {
            for (int _x = 0; _x < g1.width; ++_x) {
                g.set(_x, _y, g1.get(_x, _y) - g2.get(_x, _y));
            }
        }
        return g;
    }

    //! @brief Mult
    DEVICE_HOST friend Grid operator*(Grid const& g1, Grid const& g2) {
        Grid g(g1.width, g1.height);

        for (int _y = 0; _y < g1.height; _y++) {
            for (int _x = 0; _x < g1.width; _x++) {
                g.set(_x, _y, g1.get(_x, _y) * g2.get(_x, _y));
            }
        }
        return g;
    }
#endif // NOT_ALLOWED_BECAUSE_MIXTE_CPU_GPU_OBJECT

    // Input/Ouput
    friend ofstream& operator<<(ofstream & o, Grid const & mat) {
        if (!o)
            return(o);

        o << "Width = " << mat.width << " ";
        o << "Height = " << mat.height << " " << endl;

        if (!o)
            return(o);

        for (int _y = 0; _y < mat.height; _y++) {
            for (int _x = 0; _x < mat.width; _x++) {
                o << mat._data[_x + _y * mat.width] << " ";
            }
            o << endl;
        }
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Grid& mat) {
        char str[256];

        if (!i)
            return(i);

        mat.freeMem();

        i >> str >> str >> mat.width;
        i >> str >> str >> mat.height;

        mat.allocMem();

        for (int _y = 0; _y < mat.height; _y++) {
            for (int _x = 0; _x < mat.width; _x++) {
                i >> mat._data[_x + _y * mat.width];
            }
        }
        return i;
    }

    // C fashion Ouput
    DEVICE_HOST void printInt() {
        printf("Width = %d, Height = %d\n", width, height);
        for (int _y = 0; _y < height; _y++) {
            for (int _x = 0; _x < width; _x++) {
                (*this)[_y][_x].printInt();
                printf("  ");
            }
            printf("\n");
        }
    }

    /*! @name Globales functions specific for controling the GPU.
     * \brief Memory allocation and communication. Useful
     * for mixte utilisation.
     * @{
     */

    DEVICE_HOST void allocMem() {
        _data = alloc.allocMem(width, height, pitch);
    }

    DEVICE_HOST void freeMem() {
        alloc.freeMem(_data);
    }

    DEVICE_HOST void resize(int w, int h) {
        alloc.freeMem(_data);
        width = w;
        height = h;
        _data = alloc.allocMem(width, height, pitch);
    }

    DEVICE_HOST void gpuAllocMem() {
        _data = gpu_alloc.allocMem(width, height, pitch);
    }

    void gpuFreeMem() {
        gpu_alloc.freeMem(_data);
    }

    void gpuResize(int w, int h) {
        gpu_alloc.freeMem(_data);
        width = w;
        height = h;
        _data = gpu_alloc.allocMem(width, height, pitch);
    }

    //! HW 09/03/15 : cudaError_t cudaMemset(void* devPtr, int value, size_t count)
    //! HW 09/03/15 : Type "Node" may not be allowed in the cudaMemset2D function.
    void gpuMemSet(Node const& value) {
#ifdef CUDA_CODE
        cudaMemset2D(_data,
                     pitch,
                     value,
                     sizeof(Node) * width,
                     height);
#else
        this->resetValue(value);
#endif
    }

    //! HOST to DEVICE
    void gpuCopyHostToDevice(Grid<Node> & gpuGrid) {
#ifdef CUDA_CODE
        cudaMemcpy2D(gpuGrid._data,
                     gpuGrid.pitch,
                     _data,
                     pitch,
                     sizeof(Node) * width,
                     height,
                     cudaMemcpyHostToDevice);
#else
        // simulation
        memcpy(gpuGrid._data, _data, pitch * height);
#endif
    }

    //! DEVICE TO HOST
    void gpuCopyDeviceToHost(Grid<Node> & gpuGrid) {
#ifdef CUDA_CODE
        cudaMemcpy2D(_data,
                     pitch,
                     gpuGrid._data,
                     gpuGrid.pitch,
                     sizeof(Node) * width,
                     height,
                     cudaMemcpyDeviceToHost);
#else
        // simulation
        memcpy(_data, gpuGrid._data, pitch * height);
#endif
    }
    //! @}
};//Grid

typedef Grid<GLint> MatDensity;
typedef Grid<GLint> GridDensity;

typedef Grid<Point2D> Mat2DPoints;
typedef Grid<Point3D> Mat3DPoints;

typedef Mat2DPoints  Grid2DPoints;
typedef Mat3DPoints  Grid3DPoints;

typedef Grid<Point3D> MatPixels;
typedef Grid<Point3D> Image;

typedef Grid<GLdouble> MatObjectVal;
typedef Grid<GLfloat> MatDisparity;
typedef Grid<Point2D> MatMotion;

/*! \brief Vecteur de nodes
 */
template <class Node>
class LineOfNodes : public vector<Node> {};
template <class Node>
class SetOfLines : public vector<vector<Node> > {};

//! @}

#if TEST_CODE
//! Test program
class Test {
public:
    void run() {
        cout << "debut test Grid<> ..." << endl;
        Grid2DPoints gd(5, 5);
        cout << "... debut test GLint Grid..." << endl;
        gd.resize(10, 10);

        cout << "... debut test GLint Grid..." << endl;
        Grid<GLint> g1(10, 10), g2(10, 10), g3(10, 10);
        cout << "... debut test GLint Grid 0 ..." << endl;
        g1 = 0;
        cout << "... debut test GLint Grid 1 ..." << endl;
        g2 = g3 = 10;
        cout << "... debut test GLint Grid 2 ..." << endl;
        g1 = g2 = g3;
        cout << "... debut test GLint Grid 3 ..." << endl;
        g2 += g1 -= g3;

        cout << "... debut test Grid<Point2D> ..." << endl;

        Grid<Point2D> g4(10, 10), g5(10, 10), g6(10, 10);
        g4 = g5 = g6 = Point2D(10,10);
        g4 -= g5 - g6;
        Grid<Point2D> g7(5, 5);
        g7 += g6 - g5;

        cout << "... debut test Grid<Point3D> ..." << endl;

        // Somme of squared difference (squared l2 norm)
        Grid<Point3D> gd1(10,10), gd2(10,10);

        Grid<Point3D> ggd(5, 5);
        ggd.resize(10, 10);
        ggd = Point3D(2,3,4);
        ggd += gd1 - gd2;
        ggd += ggd * ggd;

        cout << "... debut test sum of square diff ..." << endl;
        Grid<GLfloat> gr(10, 10);
        for (int _y = 0; _y < gr.getHeight(); ++_y) {
            for (int _x = 0; _x < gr.getWidth(); ++_x) {
//                 gr.getData()[_x + _y * gr.getStride()] =
//                          ggd.getData()[_x + _y * gr.getStride()][0]
//                        + ggd.getData()[_x + _y * gr.getStride()][1]
//                        + ggd.getData()[_x + _y * gr.getStride()][2];
                gr[_y][_x] = ggd[_y][_x][0] + ggd[_y][_x][1] + ggd[_y][_x][2];
            }
        }
        cout << "... debut test ooperator [] ..." << endl;
        Grid<GLfloat> grr(10, 10);
        for (int _y = 0; _y < grr.getHeight(); ++_y) {
            for (int _x = 0; _x < grr.getWidth(); ++_x) {
                  grr[_y][_x] = ggd[_y][_x][0]
                        + ggd[_y][_x][1]
                        + ggd[_y][_x][2];
            }
        }
        cout << "fin de test Grid<> ..." << endl;
    }
};
#endif

}//namespace components

#endif // GRID_OF_NODES_H
