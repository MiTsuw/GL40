#ifndef GEOMETRY_H
#define GEOMETRY_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, H. Wang
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
//#include <vector>
//#include <boost/geometry/geometries/linestring.hpp>
//#include <boost/geometry.hpp>

#include "Node.h"

using namespace std;
using namespace components;

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

/*!
 * \defgroup FoncGeom Fonctions geometriques
 * \brief Espace de nommage geometry.
 */
/*! @{*/
namespace geometry_base
{
class Point;
class Vector;

typedef Point Point_2;
typedef Vector Vector_2;

//! \brief Classe representant un point 2D
class Point {
    double _x;
    double _y;
public:
    DEVICE_HOST Point(Point2D p) : _x(p[0]), _y(p[1]) {}

    DEVICE_HOST Point() {}
    DEVICE_HOST Point(const double x, const double y) : _x(x), _y(y) {}
    //! Acces x
    DEVICE_HOST double x() const { return _x; }
    //! Acces y
    DEVICE_HOST double y() const { return _y; }

    //! Ajout d'un vecteur ou translation de vecteur v
    DEVICE_HOST inline Point operator+(const Vector& v) const;//corps place après class Vector
    //! Soustraction vecteur ou translation de vecteur -v
    DEVICE_HOST inline Point operator-(const Vector& v) const;//corps place après class Vector
    //! Affichage
    inline friend ostream& operator<<(ostream& o, Point_2& p) {
        o << p.x() << " " << p.y() << endl;
        return o;
    }
};

//! \brief Classe representant un vecteur 2D
class Vector {
    double _x;
    double _y;
public:
    DEVICE_HOST Vector(Point2D& source, Point2D& target)
        : _x(target[0]-source[0]), _y(target[1]-source[1]) {}

    DEVICE_HOST Vector() {}
    DEVICE_HOST Vector(const double x, const double y) : _x(x), _y(y) {}
    DEVICE_HOST Vector(const Point_2& source, const Point_2& target) : _x(target.x()-source.x()), _y(target.y()-source.y()) {}
    //! Acces x
    DEVICE_HOST double x() const { return _x; }
    //! Acces y
    DEVICE_HOST double y() const { return _y; }
    //! Produit scalaire
    DEVICE_HOST double operator*(const Vector& v) const { return _x*v.x()+_y*v.y(); }
    //! Produit par un scalaire
    DEVICE_HOST Vector operator*(double d) const { return Vector(_x*d, _y*d); }
    //! Perpendiculaire sens trigo
    DEVICE_HOST Vector perpendicular() const {
      return Vector(-_y, _x);
    }
    //! Norme au carre
    DEVICE_HOST double squared_length() const {
        return _x *_x + _y *_y;
    }
};

/*!
 * \param v vecteur translation
 * \return point translate
 */
DEVICE_HOST inline Point Point::operator+(const Vector& v) const {
    return Point(_x + v.x(), _y + v.y());
}

/*!
 * \param v vecteur translation opposee
 * \return point translate
 */
DEVICE_HOST inline Point Point::operator-(const Vector& v) const {
    return Point(_x - v.x(), _y - v.y());
}

DEVICE_HOST inline Point_2 intersect(const Point_2& p0,
                         Vector_2 u,
                         const Point_2& p1,
                         Vector_2 v) {

    // Using Determinant Kramer technic
    double det = u.y()*v.x() - v.y()*u.x();

    //! HW 17/03/15 : modif
    double det_x = (u.y()*p0.x()-u.x()*p0.y())*v.x()
            - u.x()*(v.y()*p1.x()-v.x()*p1.y());

    double det_y = (u.y()*p0.x()-u.x()*p0.y())*v.y()
            - u.y()*(v.y()*p1.x()-v.x()*p1.y());

    // det is supposed != 0
    return Point_2(det_x/det, det_y/det);
}

//! JCC 23/03/15
//! To decide if p and p0 are at the same side of line p1-p2
DEVICE_HOST bool ifAtSameSideBis(const Point_2& p,
                  const Point_2& p0,
                  const Point_2& p1,
                  const Point_2& p2)
{
    Vector_2 v(p1,p2);

    v = v.perpendicular();

    double f_p1p2_p = v * Vector_2(p1, p);
    double f_p1p2_p0 = v * Vector_2(p1, p0);

    return ((f_p1p2_p * f_p1p2_p0) >= 0 ? true : false);
}

//! HW 23/03/15
//! To decide if p and p0 are at the same side of line p1-p2
DEVICE_HOST bool ifAtSameSide(const Point_2& p,
                  const Point_2& p0,
                  const Point_2& p1,
                  const Point_2& p2)
{
    double f_p1p2_p = (p.x() - p1.x()) / (p2.x() - p1.x()) - (p.y() - p1.y()) / (p2.y() - p1.y());
    double f_p1p2_p0 = (p0.x() - p1.x()) / (p2.x() - p1.x()) - (p0.y() - p1.y()) / (p2.y() - p1.y());
    return ((f_p1p2_p * f_p1p2_p0) > 0 ? true : false);
}

}//namespace geometry_base
//! @}

namespace geometry = geometry_base;

typedef geometry::Point_2 Point_2;
typedef geometry::Vector_2 Vector_2;

#endif // GEOMETRY_H
