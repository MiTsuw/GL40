/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : janvier 2015
 *
 ***************************************************************************
 */
#include "geometry_prop.h"

#define _USE_MATH_DEFINES 1
#include <math.h>

namespace geometry_prop
{
void polylineBToVecteurSegment2(const Polyline_B poly, vector<Segment_2>& vec) {
    for (int i = 0; i < poly.size(); i++) {

        Point_2 p1 = poly[i];
        if (i+1 < poly.size()) {
            Point_2 p2 = poly[i+1];
            Segment_2 s;
            if (p1.x() <= p2.x())
                s = Segment_2(p1, p2);
            else
                s = Segment_2(p2, p1);
            vec.push_back(s);
        }
    }
}//polylineBToVecteurSegment2

// sens = 0 : pivot est la roue avant, sens = 1 : roue arriere
bool intersectCercleCercle(  const int sens,
                                    const double X0,
                                    const double Y0,
                                    const double R0_carre,
                                    const double X1,
                                    const double Y1,
                                    const double R1_carre,
                                    double& X,
                                    double& Y
                                    ) {
    bool trouve = false;

    if (Y0 == Y1) {
        X = (R1_carre - R0_carre - X1*X1 + X0*X0)
                / (2 * (X0 - X1));
        //y = [2.y1 + racine ( (-2.y1)² - 4.(x1²+ x² - 2.x1.x + y1² - R1²)  )] / 2
        double delta = 4*Y1*Y1 - 4*(X1*X1 + X*X - 2*X1*X + Y1*Y1 - R1_carre);
        if (delta >= 0) {
            trouve = true;
            Y = (2*Y1 + sqrt(delta)) / 2;
        }
        else
            trouve = false;
    }
    else {
        double N = (R1_carre - R0_carre - X1*X1 + X0*X0 - Y1*Y1 + Y0*Y0)
                / (2 * (Y0-Y1));

        double A = ((X0-X1)/(Y0-Y1))*((X0-X1)/(Y0-Y1)) + 1;
        double B = 2*Y0*((X0-X1)/(Y0-Y1)) - 2*N*((X0-X1)/(Y0-Y1)) - 2*X0;
        double C = X0*X0 + Y0*Y0 + N*N - R0_carre - 2*Y0*N;

        double delta = B*B - 4*A*C;

        if (delta > 0) {
            trouve = true;
            if (sens == 0)
                X = (-B + sqrt(delta)) / (2 * A);
            else
                X = (-B - sqrt(delta)) / (2 * A);

            Y = N - X*((X0-X1)/(Y0-Y1));
        }
        else if (delta == 0) {
            trouve = true;

            X = -B / (2 * A);

            Y = N - X*((X0-X1)/(Y0-Y1));
        }
    }
    return trouve;

}//intersectCercleCercle

bool intersectCercleDroite( const int pivot,
                                    const double X0,
                                    const double Y0,
                                    const double d_carre,
                                    const double a,
                                    const double b,
                                    double& XC,
                                    double& YC
                                    ) {
    bool trouve = false;

    double A = 1 + a * a;
    double B = 2 * a * (b - Y0) - 2 * X0;
    double C = X0 * X0 + (b - Y0) * (b - Y0) - d_carre;

    double delta = B * B - 4 * A * C;

    if (delta > 0) {
        trouve = true;

        if (pivot == 0)
            XC = (-B + sqrt(delta)) / (2 * A);
        else
            XC = (-B - sqrt(delta)) / (2 * A);

        YC = a * XC + b;
    }
    else if (delta == 0) {
        trouve = true;

        XC = -B / (2 * A);

        YC = a * XC + b;
    }
    return trouve;

}//intersectCercleDroite

bool intersectVerticaleCercle(const double X0,
                                    const double Y0,
                                    const double X1,
                                    const double Y1,
                                    const double R1_carre,
                                    double& X,
                                    double& Y
                                    ) {
    bool trouve = false;

    if (R1_carre >= (X0 - X1) * (X0 - X1)) {
        trouve = true;
        X = X0;
        Y = sqrt(R1_carre - (X0 - X1) * (X0 - X1)) + Y1;
    }

    return trouve;

}//intersectVerticaleCercle

bool intersectVerticaleDroite(const double X0,
                                    const double Y0,
                                    const double a,
                                    const double b,
                                    double& XC,
                                    double& YC
                                    ) {
    bool trouve = true;

    XC = X0;

    YC = a * XC + b;

    return trouve;

}//intersectVerticaleDroite

}//namespace geometry_prop

