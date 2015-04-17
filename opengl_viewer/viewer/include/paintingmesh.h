#ifndef PAINTINGMESH_H
#define PAINTINGMESH_H

//***************************************************************************
//
// Jean-Charles CREPUT, Abdelkhalek MANSOURI
// Created in 2013, Modified in 2015
//
//***************************************************************************

#include <QtGui>
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

    explicit  PaintingMesh(QWidget *parent) :
        QGLWidget(parent)
    {
        modeColors = true;
        modeDisplay = 0;
        clearColor = Qt::black;//::white;
        lastWheel = 0;
    }
    ~PaintingMesh()
    {
        mat_points.freeMem();
        mat_colors.freeMem();
        mat_points_colonnes.freeMem();
        mat_colors_colonnes.freeMem();
        mat_points_colonnes_decalees.freeMem();
        mat_colors_colonnes_decalees.freeMem();
    }
    void initialize(ConfigParams* cp) {
        this->param = cp;
    }
    void makeObject()
    {
        components::Point3D color_default(1.0 ,1.0 ,1.0);
        ifstream fi;
        //    fi.open("./data/tsukuba_groundtruth_2.mat3dpts");
        //    fi.open("./data/tsukuba_groundtruth_2.grid3dpts");
        //    fi.open("./data/result.grid3dpts");

        string file_cfg;
        param->readConfigParameter("param_1","fileGrid3DPoints", file_cfg);

        fi.open(file_cfg.c_str());
        if (!fi) {
            std::cout << "erreur ouverture mesh" << endl;
        }
        else
            fi >> mat_points;
        fi.close();

        if (mat_points.getHeight() == 0)
            return;
        //    fi.open("./data/tsukuba_groundtruth_2.matcolors");
        //    fi.open("./data/tsukuba_groundtruth_2.gridcolors");
        //    fi.open("./data/result.gridcolors");

        param->readConfigParameter("param_1","fileGridOfColors", file_cfg);
        fi.open(file_cfg.c_str());
        if (!fi) {
            std::cout << "erreur ouverture mesh colors" << endl;
        }
        else
            fi >> mat_colors;
        fi.close();
        for (int _y = 0; _y < mat_points.getHeight(); _y++ ) {
            for (int _x = 0; _x < mat_points.getWidth(); _x++ ) {

                components::Point3D c;
                if (modeColors) {
                    c= mat_colors.get(_x,_y);
                    c.set(0,(c[0] / 255.0));
                    c.set(1,(c[1] / 255.0));
                    c.set(2,(c[2] / 255.0));

                }
                else {
                    c = color_default;
                }
                mat_colors.set(_x, _y, c);
            }
        }

        points.clear();
        colors.clear();
        for (int _y = 0; _y < mat_points.getHeight(); _y++ ) {
            for (int _x = 0; _x < mat_points.getWidth(); _x++ ) {

                components::Point3D p = mat_points.get(_x, _y);
                points.append(QVector3D(p[0],
                              p[1],
                        p[2]));
                components::Point3D c = mat_colors.get(_x, _y);
                colors.append(QVector3D(c[0],
                              c[1],
                        c[2]));
            }
        }
        cout << "Finish loading 1" << endl;

        vertices.clear();
        color_vertices.clear();
        vertices.reserve(mat_points.getHeight()*mat_points.getWidth()*6);           //////
        color_vertices.reserve(mat_points.getHeight()*mat_points.getWidth()*6);     //////

        components::Point3D p1, p2, p3, p4, p5, p6;

        nTriangles = 0;
        for (int _y = 0; _y < mat_points.getHeight()-1; _y++ ) {
            for (int _x = 0; _x < mat_points.getWidth()-1; _x++ ) {
                if (_x + 1 < mat_points.getWidth() && _y + 1 < mat_points.getHeight()) {
                    nTriangles += 2;

                    // Points
                    if (_y % 2 == 0) {

                        p1 = mat_points.get(_x, _y);
                        p2 = mat_points.get(_x + 1, _y);
                        p3 = mat_points.get(_x + 1, _y + 1);
                        p4 = mat_points.get(_x + 1, _y + 1);
                        p5 = mat_points.get(_x, _y + 1);
                        p6 = mat_points.get(_x, _y);
                    }
                    else {

                        p1 = mat_points.get(_x, _y);
                        p2 = mat_points.get(_x, _y + 1);
                        p3 = mat_points.get(_x + 1, _y);
                        p4 = mat_points.get(_x + 1, _y);
                        p5 = mat_points.get(_x, _y + 1);
                        p6 = mat_points.get(_x + 1, _y + 1);
                    }

                    vertices.append(QVector3D(p1[0], p1[1], p1[2]));
                    vertices.append(QVector3D(p2[0], p2[1], p2[2]));
                    vertices.append(QVector3D(p3[0], p3[1], p3[2]));
                    vertices.append(QVector3D(p4[0], p4[1], p4[2]));
                    vertices.append(QVector3D(p5[0], p5[1], p5[2]));
                    vertices.append(QVector3D(p6[0], p6[1], p6[2]));

                    // Couleurs
                    if (_y % 2 == 0) {

                        p1 = mat_colors.get(_x, _y);
                        p2 = mat_colors.get(_x + 1, _y);
                        p3 = mat_colors.get(_x + 1, _y + 1);
                        p4 = mat_colors.get(_x + 1, _y + 1);
                        p5 = mat_colors.get(_x, _y + 1);
                        p6 = mat_colors.get(_x, _y);
                    }
                    else {

                        p1 = mat_colors.get(_x, _y);
                        p2 = mat_colors.get(_x, _y + 1);
                        p3 = mat_colors.get(_x + 1, _y);
                        p4 = mat_colors.get(_x + 1, _y);
                        p5 = mat_colors.get(_x, _y + 1);
                        p6 = mat_colors.get(_x + 1, _y + 1);
                    }

                    color_vertices.append(QVector3D(p1[0], p1[1], p1[2]));
                    color_vertices.append(QVector3D(p2[0], p2[1], p2[2]));
                    color_vertices.append(QVector3D(p3[0], p3[1], p3[2]));
                    color_vertices.append(QVector3D(p4[0], p4[1], p4[2]));
                    color_vertices.append(QVector3D(p5[0], p5[1], p5[2]));
                    color_vertices.append(QVector3D(p6[0], p6[1], p6[2]));

                }//if
            }//for
        }//for

        mat_points_colonnes.resize(mat_points.getHeight(), mat_points.getWidth());
        mat_colors_colonnes.resize(mat_points.getHeight(), mat_points.getWidth());
        mat_points_colonnes_decalees.resize(mat_points.getHeight(), mat_points.getWidth()-1);
        mat_colors_colonnes_decalees.resize(mat_points.getHeight(), mat_points.getWidth()-1);
        for (int _y = 0; _y < mat_points.getHeight(); _y++ ) {
            for (int _x = 0; _x < mat_points.getWidth(); _x++ ) {


                components::Point3D p = mat_points.get(_x, _y);
                components::Point3D c = mat_colors.get(_x, _y);

                mat_points_colonnes.set(_y, _x, p);
                mat_colors_colonnes.set(_y, _x, c);

                if (_y % 2 == 0) {
                    if (_x < mat_points.getWidth()-1) {
                        mat_points_colonnes_decalees.set(_y, _x, p);
                        mat_colors_colonnes_decalees.set(_y, _x, c);
                    }
                }
                else {
                    if (_x > 0) {
                        mat_points_colonnes_decalees.set(_y, _x-1, p);
                        mat_colors_colonnes_decalees.set(_y, _x-1, c);
                    }
                }
            }
        }
        cout << "Finish loading 2 " << vertices.capacity() << " " << nTriangles * 3 << endl;
    }

signals:
public slots:
signals:
    void clicked();
protected:
    void initializeGL()
    {
        makeObject();
        // set up the depth value for the 3D coordinates
        glClearDepth( 1.0 );
        // enable the depth test work
        glEnable( GL_DEPTH_TEST );
        // the type of the depth test
        glDepthFunc( GL_LEQUAL );
        // make the system to revise the perspective
        glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

    }
    void paintGL()
    {

        if (modeDisplay == 0)
            displayGrid();
        else if (modeDisplay == 1)
            displayTriangles();
        else if (modeDisplay == 2)
            displayPoints();
        else if (modeDisplay == 3)
            displayLines();
    }
    void resizeGL(int width, int height)
    {

        //Set a new projection matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0,(GLdouble)width/(GLdouble)height,0.1,100.0);
        //glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0);

        glMatrixMode(GL_MODELVIEW);
        glViewport(0,0,width,height);  //Use the whole window for rendering
    }
    void mousePressEvent(QMouseEvent *event)
    {
        lastPos = event->pos();
    }
    void mouseMoveEvent(QMouseEvent *event)
    {
        int dx = event->x() - lastPos.x();
        int dy = event->y() - lastPos.y();

        if (event->buttons() & Qt::LeftButton) {
            rotateObjectBy(1 * -dy, 1 * -dx, 0);
            updateGL();
        } else if (event->buttons() & Qt::RightButton) {
            this->rotateBy(1 * -dy, 1 * dx, 0);

            updateGL();
        }
        lastPos = event->pos();
    }
    void wheelEvent(QWheelEvent *event)
    {
        int dx = event->delta();

        if (event->buttons() & Qt::RightButton) {
            rotateBy(0, 0, 1 * dx/100);
            updateGL();
        } else {

            camera.MoveForward(dx >= 0 ? -0.5 : 0.5);
            updateGL();
        }
        lastWheel = event->Wheel;
    }
    void mouseReleaseEvent(QMouseEvent * /* event */)
    {
        emit clicked();
    }


private:

    void rotateBy(int xAngle, int yAngle, int zAngle)
    {
        camera.RotateX(xAngle);
        camera.RotateY(yAngle);
        camera.RotateZ(zAngle);
        updateGL();
    }
    void rotateObjectBy(int xAngle, int yAngle, int zAngle)
    {
        camera.RotateObjectX(xAngle);
        camera.RotateObjectY(yAngle);
        updateGL();
    }
    void setClearColor(const QColor &color)
    {
        clearColor = color;
        updateGL();
    }
    void displayGrid(void)
    {
        qglClearColor(clearColor);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        camera.Render();

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);


        for (int _y = 0; _y < mat_points.getHeight(); _y++ ) {
            glVertexPointer(3, GL_FLOAT, 0, mat_points[_y]);
            glColorPointer(3, GL_FLOAT, 0, mat_colors[_y]);
            glDrawArrays(GL_LINE_STRIP, 0, mat_points.getWidth());

        }

        for (int _y = 0; _y < mat_points_colonnes.getHeight(); _y++ ) {
            glVertexPointer(3, GL_FLOAT, 0, mat_points_colonnes[_y]);
            glColorPointer(3, GL_FLOAT, 0, mat_colors_colonnes[_y]);
            glDrawArrays(GL_LINE_STRIP, 0, mat_points_colonnes.getWidth());
        }

        for (int _y = 0; _y < mat_points_colonnes_decalees.getHeight(); _y++ ) {
            glVertexPointer(3, GL_FLOAT, 0, mat_points_colonnes_decalees[_y]);
            glColorPointer(3, GL_FLOAT, 0, mat_colors_colonnes_decalees[_y]);
            glDrawArrays(GL_LINE_STRIP, 0, mat_points_colonnes_decalees.getWidth());
        }
    }
    void displayLines(void)
    {
        qglClearColor(clearColor);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        camera.Render();

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);



        for (int _y = 0; _y < mat_points.getHeight(); _y++ ) {
            glVertexPointer(3, GL_FLOAT, 0, mat_points[_y]);
            glColorPointer(3, GL_FLOAT, 0, mat_colors[_y]);

            glDrawArrays(GL_LINE_STRIP, 0, mat_points.getWidth());
        }
    }
    void displayTriangles()
    {
        qglClearColor(clearColor);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        camera.Render();

        glVertexPointer(3, GL_FLOAT, 0, vertices.constData());

        glColorPointer(3, GL_FLOAT, 0, color_vertices.constData());

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glDrawArrays(GL_TRIANGLES, 0, nTriangles*3);

    }
    void displayPoints()
    {
        qglClearColor(clearColor);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        camera.Render();

        glVertexPointer(3, GL_FLOAT, 0, points.constData());
        glColorPointer(3, GL_FLOAT, 0, colors.constData());

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glDrawArrays(GL_POINTS, 0, mat_points.getHeight() * mat_points.getWidth());
    }
    //displays the scene.
    void displayCube(void)
    {
        qglClearColor(clearColor);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        camera.Render();

        //Draw the "world" (which consists of six "nets" forming a cuboid
        glTranslatef(0.0, 0.0, -5.0);


        GLfloat size = 2.0;
        GLint LinesX = 10;
        GLint LinesZ = 10;

        GLfloat halfsize = size / 2.0;
        glColor3f(1.0,1.0,1.0);
        glPushMatrix();
        glTranslatef(0.0,-halfsize ,0.0);
        drawNet(size,LinesX,LinesZ);
        glTranslatef(0.0,size,0.0);
        drawNet(size,LinesX,LinesZ);
        glPopMatrix();
        glColor3f(0.0,0.0,1.0);
        glPushMatrix();
        glTranslatef(-halfsize,0.0,0.0);
        glRotatef(90.0,0.0,0.0,halfsize);
        drawNet(size,LinesX,LinesZ);
        glTranslatef(0.0,-size,0.0);
        drawNet(size,LinesX,LinesZ);
        glPopMatrix();
        glColor3f(1.0,0.0,0.0);
        glPushMatrix();
        glTranslatef(0.0,0.0,-halfsize);
        glRotatef(90.0,halfsize,0.0,0.0);
        drawNet(size,LinesX,LinesZ);
        glTranslatef(0.0,size,0.0);
        drawNet(size,LinesX,LinesZ);
        glPopMatrix();

        //finish rendering:
        glFlush();
        //glutSwapBuffers();
    }
    //draws a plane "net"
    void drawNet(GLfloat size, GLint LinesX, GLint LinesZ)
    {
        glBegin(GL_LINES);
        for (int xc = 0; xc < LinesX; xc++)
        {
            glVertex3f(	-size / 2.0 + xc / (GLfloat)(LinesX-1)*size,
                        0.0,
                        size / 2.0);
            glVertex3f(	-size / 2.0 + xc / (GLfloat)(LinesX-1)*size,
                        0.0,
                        size / -2.0);
        }
        for (int zc = 0; zc < LinesZ; zc++)
        {
            glVertex3f(	size / 2.0,
                        0.0,
                        -size / 2.0 + zc / (GLfloat)(LinesZ-1)*size);
            glVertex3f(	size / -2.0,
                        0.0,
                        -size / 2.0 + zc / (GLfloat)(LinesZ-1)*size);
        }
        glEnd();
    }

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
    void drawLines(QPainter *qp)
    {

        QPen pen(Qt::black, 2, Qt::SolidLine);
        qp->setPen(pen);
        qp->drawLine(20, 40, 250, 40);

        pen.setStyle(Qt::DashLine);
        qp->setPen(pen);
        qp->drawLine(20, 80, 250, 80);
    }
};



#endif // PAINTINGMESH_H
