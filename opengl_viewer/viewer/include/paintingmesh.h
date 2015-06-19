#ifndef PAINTINGMESH_H
#define PAINTINGMESH_H

//***************************************************************************
//
// Jean-Charles CREPUT, Abdelkhalek MANSOURI
// Created in 2013, Modified in 2015
//
//***************************************************************************

#include <QtGui>
#include <QKeyEvent>
#include <QWidget>
#include <QGLWidget>
#include "ConfigParams.h"
#include "camera.h"
#include <GridOfNodes.h>
#include <Node.h>

using namespace components;
class PaintingMesh : public QGLWidget
{
    Q_OBJECT
public:
    bool modeColors;
    int  modeDisplay;

    explicit  PaintingMesh(QWidget *parent);
    ~PaintingMesh();
    void initialize(ConfigParams* cp) ;
    void makeObject();

signals:
public slots:
signals:
    void clicked();
protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void mouseReleaseEvent(QMouseEvent * /* event */);
    /*////////////////////////////////////////////////////////////////////////*/
    //  Ce que j'ai ajouté
    /*////////////////////////////////////////////////////////////////////////*/
    void keyPressEvent(QKeyEvent* event);
    //*////////////////////////////////////////////////////////////////////////*/


private:

    void rotateBy(int xAngle, int yAngle, int zAngle);
    void rotateObjectBy(int xAngle, int yAngle, int zAngle);
    void setClearColor(const QColor &color);
    void displayLines(void);
    void displayTriangles();
    void displayPoints();
    void displayGrid(void);
    //displays the scene.
    void displayCube(void);
    //draws a plane "net"
    void drawNet(GLfloat size, GLint LinesX, GLint LinesZ);

    ConfigParams* param;

    CCamera camera;

    QColor clearColor;
    QPoint lastPos;
    int lastWheel;

    int nTriangles;

    // data
    Mat3DPoints mat_points;
    Mat3DPoints mat_colors;

    // buffers for opengl arrays
    QVector<QVector3D> points;
    QVector<QVector3D> colors;
    QVector<QVector3D> vertices;
    QVector<QVector3D> color_vertices;

    Mat3DPoints mat_points_colonnes;
    Mat3DPoints mat_colors_colonnes;
    Mat3DPoints mat_points_colonnes_decalees;
    Mat3DPoints mat_colors_colonnes_decalees;

public:
    void drawLines(QPainter *qp);
    void reinitCamera();
   //Fonction pour les threads
    void selfZoom(int v);
    void selfRotate(int m);
};



#endif // PAINTINGMESH_H
