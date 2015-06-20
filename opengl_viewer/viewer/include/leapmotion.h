#ifndef LEAPMOTION_H
#define LEAPMOTION_H

#include <iostream>
#include <string.h>
#include "Leap.h"


using namespace Leap;

class SampleListener : public Listener{

  public:
    SampleListener();
    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
    virtual void onFocusGained(const Controller&);
    virtual void onFocusLost(const Controller&);
    virtual void onDeviceChange(const Controller&);
    virtual void onServiceConnect(const Controller&);
    virtual void onServiceDisconnect(const Controller&);

    bool handPosLeft;
    bool handPosRight;
    bool handPosUp;
    bool handPosDown;
    bool clockwiseLock;
    bool counterClockwiseLock;
    bool swipeLtoR;
    bool swipeRtoL;
    bool keyTapReset;
};

const std::string fingerNames[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
const std::string boneNames[] = {"Metacarpal", "Proximal", "Middle", "Distal"};
const std::string stateNames[] = {"STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END"};

#endif //LEAPMOTION_H
