QT       += xml
QT       += gui
TEMPLATE = app
CONFIG += console
#CONFIG += cuda
#CONFIG += bit64
CONFIG += topo_hexa

QMAKE_LFLAGS += /INCREMENTAL:NO

#cuda:DEFINES += CUDA_CODE
topo_hexa:DEFINES += TOPOLOGIE_HEXA

unix {
        CONFIG +=
#static
        DEFINES += QT_ARCH_ARMV6
        TARGET = ../../camera/bin/application
}
win32 {
        TARGET = ../../camera/bin/application
}

SOURCES += \
    ../src/main.cpp
#../src/main.cpp

OTHER_FILES +=

LIBS += -L$$PWD/../bin/
#-llibOperators

#CUDA_SOURCES += ../src/main.cpp
#cuda_code.cu

######################################################
#
# For ubuntu, add environment variable into the project.
# Projects->Build Environment
# LD_LIBRARY_PATH = /usr/local/cuda/lib
#
######################################################

CUDA_FLOAT    = float
CUDA_ARCH     = -gencode arch=compute_20,code=sm_20
#CUDA_ARCH     = -gencode arch=compute_12,code=sm_12

win32:{

 LIBS_COMPONENTS_DIR = "C:/Users/mansouri/Desktop/project/CPU_viewer_gl40/basic_components/components/bin"
 LIBS_OPERATORS_DIR = "C:/Users/mansouri/Desktop/project/CPU_viewer_gl40/optimization_operators/operators/application/bin"

  #Do'nt use the full path.
  #Because it is include the space character,
  #use the short name of path, it may be NVIDIA~1 or NVIDIA~2 (C:/Progra~1/NVIDIA~1/CUDA/v5.0),
  #or use the mklink to create link in Windows 7 (mklink /d c:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0").
#  CUDA_DIR      = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.0"
#  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  QMAKE_LIBDIR += $$LIBS_OPERATORS_DIR $$LIBS_COMPONENTS_DIR
  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include  C:/boost_1_57_0
  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include\QtOpenGL
  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include\QtCore
  INCLUDEPATH  += C:\Qt\5.4.1\5.4\msvc2010_opengl\include\QtXml
  INCLUDEPATH  += ../include
  INCLUDEPATH  += C:\Users\BARBARA\Documents\projectGL40\CPU_viewer_gl40\optimization_operators\operators\include
  INCLUDEPATH  += C:\Users\BARBARA\Documents\projectGL40\CPU_viewer_gl40/basic_components/components/include
  INCLUDEPATH  += C:\Users\BARBARA\Documents\projectGL40\CPU_viewer_gl40/camera_test/camera/include
#$$QTDIR/include/QtOpenGL
#  LIBS         += -L$$CUDA_DIR/lib/x64 -lcuda -lcudart
  LIBS         +=  -LC:/Qt\5.4.1\5.4\msvc2010_opengl\lib -lQtGui
QMAKE_LFLAGS= -static
#-llibOperators
# -L$$CUDA_DIR/lib/Win32

# Add the necessary libraries
#  CUDA_LIBS = cuda cudart
#  NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
#  QMAKE_LFLAGS_DEBUG    = /DEBUG /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
#  QMAKE_LFLAGS_RELEASE  =         /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
}
unix:{

  ##############################################################################
  # Here to add the specific QT and BOOST paths according to your Linux system.
  # For H.W's system
  INCLUDEPATH  += ../include /usr/local/Trolltech/Qt-4.8.4/include C:/boost_1_57_0  C:\boost_1_57_0

  CUDA_DIR      = /usr/local/cuda-6.5
  QMAKE_LIBDIR += $$CUDA_DIR/lib64
  INCLUDEPATH  += $$CUDA_DIR/include C:\Qt\5.4.1\5.4\msvc2010_opengl\include

  QMAKE_CXXFLAGS += -std=c++0x

  INCLUDEPATH  += ../../../optimization_operators/operators/include
  INCLUDEPATH  += ../../../basic_components/components/include
  INCLUDEPATH  += ../../../adaptive_meshing/meshing/include
}


HEADERS += \
    ../include/TestCellular.h
