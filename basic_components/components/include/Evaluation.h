#ifndef EVALUATION_H
#define EVALUATION_H
/*
 ***************************************************************************
 *
 * Auteur : H. Wang, J.C. Creput
 * Date creation : janvier 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include "ConfigParams.h"
#include "Objectives.h"
#include "GridOfNodes.h"

using namespace std;
using namespace components;

//template <class NeuralNet, class DensityMap>
class Evaluation
{
private:
    //! Parametres globaux
    ConfigParams* params;
protected:
    //! Grid
    NeuralNet* neuralNet;
    //! Density
    DensityMap* densityMap;
    //! Objectives to evaluate
    //! Objective table
    AMObjectives objs;
    //! For smoothing term
    NeuralNet* originalMap;
public:
    Evaluation()(NeuralNet* gd,
                 DensityMap* g2d,
                 ConfigParams* par) :
    neuralNet(gd), densityMap(g2d), params(par) {}

    //! \brief Valeurs par defaut des objectifs
    void initEvaluate() {
        for (int i = 0; i < objs.dim; ++i) {
            objs.set(i, numeric_limits<double>::max());
        }
    }

    //! \brief Evaluation complete d'une solution
    void evaluate() {

        initEvaluate();

        // Compute objective
        objs.set<AMObjNames::am_k_mean>(value);
        objs.set<AMObjNames::am_length>(value);
        objs.set<AMObjNames::am_sqr_length>(value);
        objs.set<AMObjNames::am_cost>(value);
        objs.set<AMObjNames::am_sqr_cost>(value);
        objs.set<AMObjNames::am_cost_window>(value);
        objs.set<AMObjNames::am_sqr_cost_window>(value);
        objs.set<AMObjNames::am_f_smooth>(value);
        objs.set<AMObjNames::am_gd_error>(value);
    }//evaluate

    Objectives getObjectives() {
        return obj;
    }

    Objectives setObjectives(Objectives o) {
        obj = o;
    }
};

#endif // EVALUATION_H
