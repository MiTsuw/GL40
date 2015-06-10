#include "mythread.h"

/*

MyThread:: MyThread(CCamera *c, int m)
{
    this->camera=c;
    this->mode=m;
}*/

MyThread:: MyThread(PaintingMesh *p, int m)
{
    this->pm=p;
    this->mode=m;
}


void MyThread:: run()
{

    if(mode==1)
    {
        qDebug() << "Appui sur le bouton zoom";
        //camera->initCamera();
        pm->reinitCamera();


    }
}
