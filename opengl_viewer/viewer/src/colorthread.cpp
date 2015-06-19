#include "colorthread.h"

ColorThread::ColorThread(QObject *parent) : QThread(parent)
{

}

void ColorThread::updateColors(PaintingMesh *pme, bool colors)
{

    mutex.lock();
    this->pme = pme;
    this->colors = colors;
    mutex.unlock();


    if(!isRunning())
    {
        start(LowPriority);
    }
    else
    {
        restart = true;
        condition.wakeOne();
    }
}

void ColorThread::run()
{
    forever
    {
        if(this->colors)
        {
            cout<<"passage 1"<<endl;
            mutex.lock();
            pme->modeColors = true;
            pme->makeObject();
            mutex.unlock();
        }
        else
        {
            cout<<"passage 2"<<endl;
            mutex.lock();
            pme->modeColors = false;
            pme->makeObject();
            mutex.unlock();
        }


        mutex.lock();
        if (!restart)
            condition.wait(&mutex);
        restart = false;
        mutex.unlock();
    }
}

