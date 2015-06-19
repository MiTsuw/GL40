#ifndef COLORTHREAD_H
#define COLORTHREAD_H

#include <QObject>
#include <QThread>


class ColorThread : public QThread
{
public:
    ColorThread(QObject *parent = 0);
};

#endif // COLORTHREAD_H
