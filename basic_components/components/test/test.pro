QT       += xml
QT       -= gui
TEMPLATE = app
CONFIG += console
CONFIG += cuda
#CONFIG += static

cuda:DEFINES += CUDA_CODE

unix {
        CONFIG +=
#static
        DEFINES += QT_ARCH_ARMV6
        TARGET = ../../components/bin/application
}
win32 {
        TARGET = ../../components/bin/application
}

SOURCES +=
# ../src/main.cpp

INCLUDEPATH  += ../include  C:/boost_1_55_0

LIBS += -L$$PWD/../bin/
#-llibComponents

CUDA_SOURCES += ../src/main.cu
#cuda_code.cu

######################################################
#
# For ubuntu, add environment variable into the project.
# Projects->Build Environment
# LD_LIBRARY_PATH = /usr/local/cuda/lib
#
######################################################

CUDA_FLOAT    = float
#CUDA_ARCH     = -gencode arch=compute_20,code=sm_20
CUDA_ARCH     = -gencode arch=compute_12,code=sm_12


win32:{
  LIBS_COMPONENTS_DIR = "D:/creput/AUT14/gpu_work/popip/basic_components/components/bin"

  #Do'nt use the full path.
  #Because it is include the space character,
  #use the short name of path, it may be NVIDIA~1 or NVIDIA~2 (C:/Progra~1/NVIDIA~1/CUDA/v5.0),
  #or use the mklink to create link in Windows 7 (mklink /d c:\cuda "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0").
#  CUDA_DIR      = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.0"
  CUDA_DIR      = "C:/Progra~1/NVIDIA~2/CUDA/v5.0"
#  QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
  QMAKE_LIBDIR += $$CUDA_DIR/lib/Win32 $$LIBS_COMPONENTS_DIR
  INCLUDEPATH  += $$CUDA_DIR/include D:/creput/AUT14/gpu_work/popip/basic_components/components/include $$CUDA_DIR/include C:/QT/qt-everywhere-opensource-src-4.8.4/include  C:/Qt/qt-everywhere-opensource-src-4.8.4/include/QtOpenGL ../include C:/boost_1_55_0

#$$QTDIR/include/QtOpenGL
#  LIBS         += -L$$CUDA_DIR/lib/x64 -lcuda -lcudart
  LIBS         += -lcuda -lcudart
#-llibComponents
# -L$$CUDA_DIR/lib/Win32

# Add the necessary libraries
#  CUDA_LIBS = cuda cudart
#  NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
  QMAKE_LFLAGS_DEBUG    = /DEBUG /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
#  QMAKE_LFLAGS_RELEASE  =         /NODEFAULTLIB:libc.lib /NODEFAULTLIB:libcmt.lib
}
unix:{
  CUDA_DIR      = /usr/local/cuda
  QMAKE_LIBDIR += $$CUDA_DIR/lib
  INCLUDEPATH  += $$CUDA_DIR/include
  LIBS += -lcudart -lcuda
  QMAKE_CXXFLAGS += -std=c++0x
}

DEFINES += "CUDA_FLOAT=$${CUDA_FLOAT}"

NVCC_OPTIONS = --use_fast_math -DCUDA_FLOAT=$${CUDA_FLOAT}
cuda:NVCC_OPTIONS += -DCUDA_CODE
#NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

QMAKE_EXTRA_COMPILERS += cuda

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CONFIG(release, debug|release) {
  OBJECTS_DIR = ./release
  cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}
CONFIG(debug, debug|release) {
  OBJECTS_DIR = ./debug
  cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine 32 $$CUDA_ARCH -c -D_DEBUG -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
}

#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
