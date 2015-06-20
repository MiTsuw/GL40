#include "mythread.h"



void MyThread:: run()
{

    while(true)
    {
        QMutex mutex;
        // prevent other threads from changing the "Stop" value
        mutex.lock();
        if(this->Stop) break;
        mutex.unlock();
        //qDebug()<<"speed"<<speed<<endl;
        // emit the signal update screen
        if(type==0)
        {
            //qDebug()<<"mode zoom"<<speedZoom<<endl;
            emit updateScreen(speedZoom);
        }
        else if(type==1)
        {
           // qDebug()<<"mode rotation"<<modeRotation<<endl;
            emit updateScreen(modeRotation);
        }
        else if(type==2)
        {
           // qDebug()<<"mode rotation"<<modeRotation<<endl;
            emit lmOnFrame();
        }


        // slowdown the rotation, msec
        this->msleep(50);

    }
}

void MyThread::updateSpeed(int s)
{
    this->speedZoom=s;
}

void MyThread::updateModeRotation(int m)
{
    this->modeRotation=m;
}

void MyThread:: setType(int t)
{
    this->type=t;
}

