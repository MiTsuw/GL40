#ifndef MYTHREAD
#define MYTHREAD

#include <QThread>
#include "paintingmesh.h"
#include <QGLWidget>


class MyThread : public QThread
{
     Q_OBJECT
private:
    CCamera *camera;
    PaintingMesh* pm;
    int mode;
public:
    //MyThread(CCamera *c, int m);
    MyThread(PaintingMesh *p, int m);
protected:
     void run();
};













#endif // MYTHREAD

