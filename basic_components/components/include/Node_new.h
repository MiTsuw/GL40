#ifndef NODE_H
#define NODE_H

//***************************************************************************
//
// J.C. Creput, janvier 2015
//
//***************************************************************************
#include <iostream>
#include <fstream>
#include <vector>
#include "lib_global.h"
#include "macros_cuda.h"

#define TEST_CODE 0

using namespace std;

typedef short GLshort;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;

typedef GLint GrayValue;
typedef GLdouble IntensityValue;

#include <cstddef>
#include <boost/geometry/geometries/point.hpp>

namespace components
{

/*!
 * \defgroup Node
 * \brief Espace de nommage components
 * Il comporte les nodes (point, neurone)
 */
/*! @{*/

/*!
 * \brief Basic point class, having coordinates defined in a neutral way (idem Boost)
 */
template<typename TypeCoordinate, std::size_t Dimension>
class Point
{
protected:
    TypeCoordinate _value[Dimension];
public:
    //! @brief Default constructor
    DEVICE_HOST inline Point() {}

    DEVICE_HOST inline Point(Point const& p2) {
        if (Dimension >= 1)
            _value[0] = p2._value[0];
        if (Dimension >= 2)
            _value[1] = p2._value[1];
        if (Dimension >= 3)
            _value[2] = p2._value[2];
    }

    //! @brief Affectation
    DEVICE_HOST Point& operator=(Point const& p2) {
        if (Dimension >= 1)
            _value[0] = p2._value[0];
        if (Dimension >= 2)
            _value[1] = p2._value[1];
        if (Dimension >= 3)
            _value[2] = p2._value[2];
        return *this;
    }

    //! @brief Constructor
    DEVICE_HOST explicit inline Point(TypeCoordinate const& v0, TypeCoordinate const& v1 = 0, TypeCoordinate const& v2 = 0) {
        if (Dimension >= 1)
            _value[0] = v0;
        if (Dimension >= 2)
            _value[1] = v1;
        if (Dimension >= 3)
            _value[2] = v2;
    }

    //! @brief Constructor
    DEVICE_HOST explicit inline Point(TypeCoordinate const& v0) {
        if (Dimension >= 1)
            _value[0] = v0;
        if (Dimension >= 2)
            _value[1] = v0;
        if (Dimension >= 3)
            _value[2] = v0;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate& operator[](std::size_t const i) {
        return _value[i];
    }

    //! @brief Get coordinate
    template <std::size_t K>
    DEVICE_HOST inline TypeCoordinate const& get() const {
        return _value[K];
    }

    //! @brief Set coordinate
    template <std::size_t K>
    DEVICE_HOST inline void set(TypeCoordinate const& value) {
        _value[K] = value;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate const& get(std::size_t const i) const {
        return _value[i];
    }

    //! @brief Set coordinatev for loop only
    DEVICE_HOST inline void set(std::size_t const i, TypeCoordinate const& value) {
        _value[i] = value;
    }
    DEVICE_HOST inline Point& operator+=(Point& p2) {
        if (Dimension >= 1)
            _value[0] += p2._value[0];
        if (Dimension >= 2)
            _value[1] += p2._value[1];
        if (Dimension >= 3)
            _value[2] += p2._value[2];
        return *this;
    }
    DEVICE_HOST inline Point& operator-=(Point& p2) {
        if (Dimension >= 1)
            _value[0] -= p2._value[0];
        if (Dimension >= 2)
            _value[1] -= p2._value[1];
        if (Dimension >= 3)
            _value[2] -= p2._value[2];
        return *this;
    }
    DEVICE_HOST inline Point& operator*=(Point& p2) {
        if (Dimension >= 1)
            _value[0] *= p2._value[0];
        if (Dimension >= 2)
            _value[1] *= p2._value[1];
        if (Dimension >= 3)
            _value[2] *= p2._value[2];
        return *this;
    }

    DEVICE_HOST inline friend Point operator+(const Point& p1, const Point& p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] + p2._value[0];
        if (Dimension >= 2)
            p._value[1] = p1._value[1] + p2._value[1];
        if (Dimension >= 3)
            p._value[2] = p1._value[2] + p2._value[2];
        return p;
    }

    DEVICE_HOST inline friend Point operator-(const Point& p1, const Point& p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] - p2._value[0];
        if (Dimension >= 2)
            p._value[1] = p1._value[1] - p2._value[1];
        if (Dimension >= 3)
            p._value[2] = p1._value[2] - p2._value[2];
        return p;
    }

    //! Scalar product
    DEVICE_HOST inline friend GLfloat operator*(const Point& p1, const Point& p2) {
        GLfloat f = 0.0;
        if (Dimension >= 1)
            f += p1._value[0] * p2._value[0];
        if (Dimension >= 2)
            f += p1._value[1] * p2._value[1];
        if (Dimension >= 3)
            f += p1._value[2] * p2._value[2];
        return f;
    }

    DEVICE_HOST inline friend Point operator*(const Point& p1, const GLfloat& p2) {
        Point p;
        if (Dimension >= 1)
            p._value[0] = p1._value[0] * p2;
        if (Dimension >= 2)
            p._value[1] = p1._value[1] * p2;
        if (Dimension >= 3)
            p._value[2] = p1._value[2] * p2;
        return p;
    }

    DEVICE_HOST inline void printInt() {
        if (Dimension >= 1)
            printf("%d ", (int)_value[0]);
        if (Dimension >= 2)
            printf("%d ", (int)_value[1]);
        if (Dimension >= 3)
            printf("%d ", (int)_value[2]);
    }

    //! Scalar product
    DEVICE_HOST inline friend GLfloat fabs(const Point& p) {
        GLfloat f = 0.0f;
        if (Dimension >= 1)
            f += fabs(p._value[0]);
        if (Dimension >= 2)
            f += fabs(p._value[1]);
        if (Dimension >= 3)
            f += fabs(p._value[2]);
        return f;
    }

};//Point

typedef Point<GLint, 1> Point1DInt;
typedef Point<GLfloat, 1> Point1D;
typedef Point<GLint, 2> PointCoord;

class Point2D : public Point<GLfloat, 2> {
public:
    //! Constructeurs
    DEVICE_HOST inline Point2D() : Point() {}
    DEVICE_HOST inline Point2D(Point2D const& p) : Point(p){}
    DEVICE_HOST explicit inline Point2D(GLfloat const& v0,
                                        GLfloat const& v1 = 0) : Point(v0, v1) {}

    //! @brief Affectation
    DEVICE_HOST Point2D& operator=(Point2D const& p2) {
        Point::operator=(p2);//((Point&)*this) = p2;//
        return *this;
    }

    friend ofstream& operator<<(ofstream& o, Point2D& p) {
        o << p._value[0] << "  " << p._value[1] << "  ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Point2D& p) {
        i >> p._value[0] >> p._value[1];
        return i;
    }
    DEVICE_HOST inline friend Point2D operator+(Point2D const& p1, Point2D const& p2) {
        Point2D p;
        (Point&) p = (Point&) p1 + (Point&) p2;
        return p;
    }
    DEVICE_HOST inline friend Point2D operator-(Point2D const& p1, Point2D const& p2) {
        Point2D p;
        (Point&) p = (Point&) p1 - (Point&) p2;
        return p;
    }

    //! Scalar product
    DEVICE_HOST inline friend GLfloat operator*(Point2D const& p1, Point2D const& p2) {
        GLfloat p;
        p = (Point&) p1 * (Point&) p2;
        return p;
    }
};

class Point3D : public Point<GLfloat, 3> {
public:
    //! Constructeurs
    DEVICE_HOST    inline Point3D() : Point() {}
    DEVICE_HOST    inline Point3D(Point3D const& p) : Point(p){}
    DEVICE_HOST    explicit inline Point3D(GLfloat const& v0,
                                           GLfloat const& v1 = 0,
                                           GLfloat const& v2 = 0) : Point(v0, v1, v2) {}

    //! @brief Affectation
    DEVICE_HOST    Point3D& operator=(Point3D const& p2) {
        (Point&) *this = p2;//Point::operator=(p2);
        return *this;
    }

    DEVICE_HOST inline friend Point3D operator+(Point3D const& p1, Point3D const& p2) {
        Point3D p;
        (Point&) p = (Point&) p1 + (Point&) p2;
        return p;
    }

    DEVICE_HOST inline friend Point3D operator-(Point3D const& p1, Point3D const& p2) {
        Point3D p;
        (Point&) p = (Point&) p1 - (Point&) p2;
        return p;
    }

    //! Scalar product
    DEVICE_HOST inline friend GLfloat operator*(Point3D const& p1, Point3D const& p2) {
        GLfloat p;
        p = (Point&) p1 * (Point&) p2;
        return p;
    }

    friend ofstream& operator<<(ofstream& o, Point3D& p) {
        o << p._value[0] << "  " << p._value[1] << "  " << p._value[2] << "  ";
        return o;
    }

    friend ifstream& operator>>(ifstream& i, Point3D& p) {
        i >> p._value[0] >> p._value[1] >> p._value[2];
    }
};

//! @}

#if TEST_CODE
//! Test program
class Test {
public:
    void run() {
        cout << "... begin test ..." << endl;

        Point1D p1(10), p2(15), p3;
        p3 = p1 + p2;
        p3 = p1 - p2;
        Point1D p4 = p3;//cons copie
        Point1D p5 = p4;
        cout << "point p = " << p5.get<0>() << endl;

        Point2D pp1(10, 15), pp2(15, 20), pp3;
        pp3 = pp1;
        Point2D pp;
        pp = pp1 + pp2;
        pp2 = pp1 - pp2;
        pp3 = pp1 * pp3;
        Point2D pp4 = pp3;//cons copie
        Point2D pp5 = pp2;
        cout << "point p = (" << pp5.get<0>() << ", " <<  pp5.get<1>() << ")" << endl;
        pp5 = pp4;
        cout << "point p = (" << pp5.get<0>() << ", " <<  pp5.get<1>() << ")" << endl;

        Point3D ppp1(10, 15, 20), ppp2(15, 20, 25), ppp3;
        ppp3 = ppp1;
        Point3D ppp;
        ppp = ppp1 + ppp2;
        ppp2 = ppp1 - ppp2;
        ppp3 = ppp1 * ppp3;
        Point3D ppp4 = ppp3;//cons copie
        Point3D ppp5 = ppp2;
        cout << "point p = (" << ppp5.get<0>() << ", " <<  ppp5.get<1>()  << ", " <<  ppp5.get<2>() << ")" << endl;
        ppp5 = ppp4;
        cout << "point p = (" << ppp5.get<0>() << ", " <<  ppp5.get<1>()  << ", " <<  ppp5.get<2>() << ")" << endl;
        cout << "... end test ..." << endl << endl;
    }
};
#endif

}//namespace components


#endif // NODE_H
