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

        // emit the signal update screen
        emit updateScreen();

        // slowdown the rotation, msec
        this->msleep(50);

    }
}
