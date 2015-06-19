#ifndef COLORTHREAD_H
#define COLORTHREAD_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <paintingmesh.h>


class ColorThread : public QThread
{
private:
    QMutex mutex;
    QWaitCondition condition;
    bool restart;

    PaintingMesh *pme;
    bool colors;

protected:
   void run();

public:
    ColorThread(QObject *parent = 0);
    void updateColors(PaintingMesh *pme, bool colors);
};

#endif // COLORTHREAD_H
