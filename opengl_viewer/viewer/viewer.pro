#-------------------------------------------------
#
# Project created by QtCreator 2013-01-09T01:18:19
#
#-------------------------------------------------

#greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

QMAKE_LFLAGS += /INCREMENTAL:NO

QT           += core gui opengl widgets
TARGET        = ../bin/viewer
TEMPLATE      = app

CONFIG += static
CONFIG += console

SOURCES      += \
    src/main.cpp \
    src/camera.cpp

HEADERS      += \
    include/paramframe.h \
    include/paintingmesh.h \
    include/ctrlwidget.h \
    include/camera.h \
    include/interfaceUI.h \

FORMS        +=


OTHER_FILES  +=

######################################################
#
# For ubuntu, add environment variable into the project.
# Projects->Build Environment
# LD_LIBRARY_PATH = /usr/local/cuda/lib
#
######################################################



win32:{


  INCLUDEPATH  +=  C:\Qt\qt-everywhere-opensource-src-5.4.1\include C:\QT\qt-everywhere-opensource-src-5.4.1\include\QtOpenGL
  INCLUDEPATH  += include ../../basic_components/components/include

}

unix:{

  ##############################################################################
  # Here to add the specific QT and BOOST paths according to your Linux system.
  # For H.W's system
  INCLUDEPATH  += ../include   C:/boost_1_57_0

  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include


  QMAKE_CXXFLAGS += -std=c++0x

  INCLUDEPATH  += ../../basic_components/components/include
  INCLUDEPATH  += include
}


