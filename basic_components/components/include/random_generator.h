#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
/*!
 * \defgroup RandomGen Generateur de nombre aleatoire
 * \brief initialisation, generation d'entier (int),
 * generation de flottant (double).
 * @{
 */
//!
inline void aleat_initialize(unsigned int seed)
{
    srand (seed);
}

inline void aleat_initialize(void)
{
    srand (time (NULL));
}

inline int aleat_int(double a, double b)
{
    double ret = ( (double)rand()/(double)RAND_MAX * (b-a) + a );
    int ret_2 = (int) ret;
    return ret_2;

}

inline double aleat_double(double a, double b) {
    double ret = ( (double)rand()/(double)RAND_MAX * (b-a) + a );
    return ret;
}

inline float aleat_float(float a, float b) {
    double ret = ( (float)rand()/(float)RAND_MAX * (b-a) + a );
    return ret;
}
//! @}

#endif // RANDOM_GENERATOR_H
