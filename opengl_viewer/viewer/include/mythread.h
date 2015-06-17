#ifndef MYTHREAD
#define MYTHREAD

#include <QThread>
#include "paintingmesh.h"
#include <QGLWidget>


class MyThread : public QThread
{
     Q_OBJECT
private:
    int mode;
    QMutex mutex;

public:
    bool Stop;
    //MyThread(CCamera *c, int m);
    MyThread(QObject *parent = 0, bool stop=false): QThread(parent), Stop(stop)
    {

    }

protected:
     void run();

signals:
     void updateScreen();
};


#endif // MYTHREAD
