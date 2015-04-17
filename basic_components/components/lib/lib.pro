
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

#Si win32-msvc2010 ou win32-g++ (minGW)
win32 {
        win32-g++:QMAKE_CXXFLAGS += -msse2 -mfpmath=sse
        TARGET = ../../components/bin/libComponents
}

#Si linux-arm-gnueabi-g++ pour cross-compile vers linux et/ou raspberry pi
unix {
        CONFIG += shared
        QMAKE_CXXFLAGS +=
        DEFINES += QT_ARCH_ARMV6
        TARGET = ../../components/bin/libComponents
}

TEMPLATE = lib
arm {
    INCLUDEPATH  += ../include /home/pi/boost_1_57_0
}
else {
    INCLUDEPATH  += ../include C:\boost_1_57_0
}
#utile pour QT librairie export
DEFINES += LIB_LIBRARY

SOURCES +=

HEADERS +=\
    ../include/ConfigParams.h \
    ../include/random_generator.h \
    ../include/GridOfNodes.h \
    ../include/Node.h \
    ../include/Objectives.h \
    ../include/distances_matching.h \
    ../include/GridPatch.h \
    ../include/filters.h \
    ../include/Converter.h \
    ../include/binary_operations.h \
    ../include/macros_cuda.h \
    ../include/ViewGrid.h \
    ../include/geometry.h \
    ../include/Cell.h \
    ../include/SpiralSearch.h \
    ../include/NeuralNet.h \
    ../include/Trace.h \
    ../include/ImageRW.h \
    ../include/NIter.h

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

win32:{

  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include
  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include\QtOpenGL
  INCLUDEPATH  +=  ../include
  INCLUDEPATH  +=  C:/boost_1_57_0

}
unix:{

  QMAKE_CXXFLAGS += -std=c++0x
}



