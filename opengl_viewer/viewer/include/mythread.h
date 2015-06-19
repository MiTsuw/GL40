#ifndef MYTHREAD
#define MYTHREAD

#include <QThread>
#include "paintingmesh.h"
#include <QGLWidget>


class MyThread : public QThread
{
     Q_OBJECT
private:
    QMutex mutex;
    int speedZoom;
    int type; //0: ZOOM     1: Rotation
    int modeRotation; // 0: rotation à gauche   1: rotation à droite

public:
    bool Stop;
    //MyThread(CCamera *c, int m);
    MyThread(QObject *parent = 0,int t=0, bool stop=false): QThread(parent), Stop(stop)
    {
        type=t;
        speedZoom=1;
        type= 0;
        modeRotation=0;
    }


    void setType(int t);
protected:
     void run();
public slots:
     void updateSpeed(int);
     void updateModeRotation(int);

signals:
     void updateScreen(int);


};













#endif // MYTHREAD

