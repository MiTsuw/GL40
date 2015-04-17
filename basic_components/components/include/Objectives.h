#ifndef OBJECTIVES_H
#define OBJECTIVES_H
/*
 ***************************************************************************
 *
 * Author : H. Wang, J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include "ConfigParams.h"
#include "macros_cuda.h"
#include "Node.h"

using namespace std;

namespace components
{

template<typename TypeCoordinate, std::size_t Dimension>
class Objectives;

enum AMObjNames { distr, length, sqr_length, cost, sqr_cost, cost_window, sqr_cost_window, smooth, gd_error };
typedef Objectives<GLfloat, 9> AMObjectives;

template<typename TypeCoordinate, std::size_t Dimension>
class Objectives
{
protected:
    TypeCoordinate _objectives[Dimension];
    TypeCoordinate _weights[Dimension];
    int dim;
public:
    /*! @name Objectifs et criteres du probleme
     * @{
     */

    //! @brief Default constructor
    inline Objectives() : dim(Dimension) {}

    //! @brief Constructor
    explicit inline Objectives(TypeCoordinate const& v0,
                               TypeCoordinate const& v1 = 0,
                               TypeCoordinate const& v2 = 0)
        : dim(Dimension) {

        if (Dimension >= 1)
            _objectives[0] = v0;
        if (Dimension >= 2)
            _objectives[1] = v1;
        if (Dimension >= 3)
            _objectives[2] = v2;
    }

    //! @brief Get coordinate for loop only
    DEVICE_HOST inline TypeCoordinate& operator[](std::size_t const i) {
        return _objectives[i];
    }

    //! @brief Get coordinate
    template <std::size_t K>
    inline TypeCoordinate const& get() const {
        return _objectives[K];
    }

    //! @brief Set coordinate
    template <std::size_t K>
    inline void set(TypeCoordinate const& value) {
        _objectives[K] = value;
    }
    //! @brief Get coordinate for loop only
    inline TypeCoordinate const& get(std::size_t const i) const {
        return _objectives[i];
    }

    //! @brief Set coordinatev for loop only
    inline void set(std::size_t const i, TypeCoordinate const& value) {
        _objectives[i] = value;
    }
    template <std::size_t K>
    inline TypeCoordinate const& get_weights() const {
        return _weights[K];
    }

    //! @brief Set coordinate
    template <std::size_t K>
    inline void set_weights(TypeCoordinate const& value) {
        _weights[K] = value;
    }
    //! @brief Get coordinate for loop only
    inline TypeCoordinate const& get_weights(std::size_t const i) const {
        return _weights[i];
    }

    //! @brief Set coordinatev for loop only
    inline void set_weights(std::size_t const i, TypeCoordinate const& value) {
        _weights[i] = value;
    }
    //! @}

    /*!
     * \return valeur de la fonction objectif agregative
     */
    TypeCoordinate computeObjectif(void) {

        TypeCoordinate objectif;

        objectif = 0;
        for (int i = 0; i < dim; ++i) {
            objectif += get(i) * get_weights(i);
        }

        return objectif;
    }

    /*!
     * \param best solution comparee
     * \return vrai si objectif de l'appelant (ie la solution courante) est inferieur ou egal a celui de la solution comparee
     */
    bool isBest(Objectives* best) {
        bool res = false;

        if (computeObjectif() <= best->computeObjectif())
            res = true;

        return res;
    }

    /*!
     * \return vrai si solution admissible
     */
    bool isSolution() {
        bool ret = true;
        for (int i = 0; i < dim; ++i) {
            if (get(i) > 0) {
                ret = false;
                break;
            }
        }
        return ret;
    }//isSolution
};

}//namespace components

#endif // OBJECTIVES_H
