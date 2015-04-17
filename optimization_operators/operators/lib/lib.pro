#-------------------------------------------------
#
# Project created by QtCreator 2014-05-17T18:44:07
#
#-------------------------------------------------

QMAKE_LFLAGS += /INCREMENTAL:NO

QT       += xml
QT       -= gui

CONFIG += console
CONFIG += exceptions rtti

INCLUDEPATH  += ../../../basic_components/components/include
#Si win32-msvc2010 ou win32-g++ (minGW)
win32 {
        win32-g++:QMAKE_CXXFLAGS += -msse2 -mfpmath=sse
        TARGET = ../../application/bin/libApplication
}
#Si linux-arm-gnueabi-g++ pour cross-compile vers linux et/ou raspberry pi
unix {
        CONFIG += shared
        QMAKE_CXXFLAGS +=
        DEFINES += QT_ARCH_ARMV6
        TARGET = ../../application/bin/libApplication
}

TEMPLATE = lib

arm {
    INCLUDEPATH  += ../include /home/pi/boost_1_57_0
}
else {
    INCLUDEPATH  += ../include C:/boost_1_57_0
}

#utile pour QT librairie export
DEFINES += LIB_LIBRARY

cgal:LIBS += -LD:\CGAL-4.3_NMAKE_RELEASE\lib -LC:\boost_1_57_0\lib32-msvc-10.0

SOURCES +=

HEADERS +=\
    ../include/random_generator.h \
    ../include/CellularMatrix.h \
    ../include/distance_functors.h \
    ../include/adaptator_basics.h

cgal:HEADERS +=

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}


######################################################
#
# For ubuntu, add environment variable into the project.
# Projects->Build Environment
# LD_LIBRARY_PATH = /usr/local/cuda/lib
#
######################################################

CUDA_FLOAT    = float
CUDA_ARCH     = -gencode arch=compute_20,code=sm_20

win32:{

  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include
  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include\QtOpenGL
  INCLUDEPATH  +=  ../include
  INCLUDEPATH  +=  C:/boost_1_57_0

}
unix:{

  QMAKE_CXXFLAGS += -std=c++0x
}


