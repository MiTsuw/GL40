#ifndef CTRLWIDGET_H
#define CTRLWIDGET_H
//***************************************************************************
//
// Jean-Charles CREPUT, Abdelkhalek MANSOURI
// Created in 2013, Modified in 2015
//
//***************************************************************************

#include <QWidget>
#include "ConfigParams.h"
#include "Converter.h"
#include "interfaceUI.h"
#include <iostream>
#include <ctime>

using namespace std;

#define CPU 0
#define GPU 1
#define GPU_NAY 0

//#if GPU
extern "C" void cuda_main(ConfigParams* cp);
//#endif

//#if GPU_NAY
extern "C" void cuda_main_nay( int, int , float * );
//#endif

// DIRECTIVES DE COMPILATION
#define ANCIEN				0

#define SPIRAL_SEARCH       1
#define SPIRAL_SEARCH_REFRESH_RATE  4000
// to choose whether lissage or not
#define LISSAGE             1

#define BREAK_ITE			10000

#define DEMO    			0

//extern TMatPix* MatRess;

static long cptIte; // nombre d'iterations
static long maxIte; // nombre total d'iterations
static int breakIte; // nombre d'iteration avant affichage

namespace Ui {
class MainWindow;
}

class CtrlWidget : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit CtrlWidget(QWidget *parent, char* cfgFile);
    ~CtrlWidget();

private:
    Ui::MainWindow *ui;
    Converter convert;
    ConfigParams* param;

};
#endif // CTRLWIDGET_H
