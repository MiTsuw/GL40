#ifndef LIB_GLOBAL_H
#define LIB_GLOBAL_H
/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : janvier 2015
 *
 ***************************************************************************
 */

#include <QtCore/qglobal.h>

#if defined(LIB_LIBRARY)
#  define LIBSHARED_EXPORT Q_DECL_EXPORT
#else
#  define LIBSHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // LIB_GLOBAL_H
