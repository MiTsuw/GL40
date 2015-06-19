#ifndef PARAMFRAME_H
#define PARAMFRAME_H
//***************************************************************************
//
// Jean-Charles CREPUT, Abdelkhalek MANSOURI
// Created in 2013, Modified in 2015
//
//***************************************************************************

#include <QWidget>
#include <QSlider>
#include "paintingmesh.h"
#include <QGroupBox>
#include <QHeaderView>
#include <QRadioButton>
#include <QApplication>
#include <QLabel>
#include <QHBoxLayout>

#include <QApplication>
#include <QPushButton>
#include "mythread.h"

QT_BEGIN_NAMESPACE
class QGroupBox;
class QRadioButton;
QT_END_NAMESPACE

class ParamFrame : public QWidget
{
    Q_OBJECT

private:
    QGroupBox *twoSidedGroupBox;
    QGroupBox *zoomGroupBox;
    QGroupBox *rotationGroupBox;

    QGroupBox *colorsGroupBox;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_0;

    QPushButton *view2DEnabledButton;
    QPushButton *view2DDisabledButton;
    QPushButton *colorsEnabledButton;
    QPushButton *colorsDisabledButton;

    QGroupBox *displayGroupBox;
    QPushButton *displayMButton;
    QPushButton *displayTButton;
    QPushButton *displayPButton;
    QPushButton *displayLButton;
    QLabel *label;
    QHBoxLayout *horizontalLayout_1;
    QHBoxLayout *horizontalLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QHBoxLayout *horizontalLayout_4;
    QGridLayout *gridLayout_2;
    PaintingMesh *pme;

    QIcon *playIcon;
    QIcon *stopIcon;

    //Déclarations des boutons et du thread zoom
    QPushButton* btnStartZoom;
    QPushButton* btnStopZoom;
    MyThread* tZoomCamera;

    //Déclarations des boutons et du thread rotation
    QPushButton* btnStartRotation;
    QPushButton* btnStopRotation;
    MyThread* tRotateCamera;

    QSlider* sliderZoom;
    QSlider* sliderRotation;

    QPushButton *btnRefresh;



public:
    explicit ParamFrame(QFrame *parent = 0);
    ~ParamFrame();

    void setWidgetsLink( PaintingMesh *);

private slots:

    void updateView();
    void updateDisplay();

    //Fonction pour le thread zoom
    void autoSelfZoom(int v);
    void startZoom();
    void stopZoom();

    //Fonction pour le thread rotatecamera
    void autoSelfRotate(int m);
    void startRotate();
    void stopRotate();



    //Fonction pour rafraichir l'image
    void refreshCamera();

};

#endif // PARAMFRAME_H
