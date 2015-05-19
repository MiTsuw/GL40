#include "camera.h"
#include "math.h"
#include <iostream>
#include <QDebug>
//# include <Windows.h>

#define SQR(x) (x*x)

#define NULL_VECTOR F3dVector(0.0f,0.0f,0.0f)

SF3dVector F3dVector ( GLfloat x, GLfloat y, GLfloat z )
{
    SF3dVector tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    return tmp;
}

GLfloat GetF3dVectorLength( SF3dVector * v)
{
    return (GLfloat)(sqrt(SQR(v->x)+SQR(v->y)+SQR(v->z)));
}

SF3dVector Normalize3dVector( SF3dVector v)
{
    SF3dVector res;
    float l = GetF3dVectorLength(&v);
    if (l == 0.0f) return NULL_VECTOR;
    res.x = v.x / l;
    res.y = v.y / l;
    res.z = v.z / l;
    return res;
}

SF3dVector operator+ (SF3dVector v, SF3dVector u)
{
    SF3dVector res;
    res.x = v.x+u.x;
    res.y = v.y+u.y;
    res.z = v.z+u.z;
    return res;
}
SF3dVector operator- (SF3dVector v, SF3dVector u)
{
    SF3dVector res;
    res.x = v.x-u.x;
    res.y = v.y-u.y;
    res.z = v.z-u.z;
    return res;
}


SF3dVector operator* (SF3dVector v, float r)
{
    SF3dVector res;
    res.x = v.x*r;
    res.y = v.y*r;
    res.z = v.z*r;
    return res;
}

SF3dVector CrossProduct (SF3dVector * u, SF3dVector * v)
{
    SF3dVector resVector;
    resVector.x = u->y*v->z - u->z*v->y;
    resVector.y = u->z*v->x - u->x*v->z;
    resVector.z = u->x*v->y - u->y*v->x;

    return resVector;
}
float operator* (SF3dVector v, SF3dVector u)	//dot product
{
    return v.x*u.x+v.y*u.y+v.z*u.z;
}

CCamera::CCamera()
{
    //Init with standard OGL values:

    Position = F3dVector (0.0, 0.0,	0.0);
    ViewDir = F3dVector( 0.0, 0.0, -1.0);
    RightVector = F3dVector (1.0, 0.0, 0.0);
    UpVector = F3dVector (0.0, 1.0, 0.0);

    target = F3dVector (0.0, 0.0, -10.0);

    //Only to be sure:
    RotatedX = RotatedY = RotatedZ = 0.0;
}

void CCamera::initCamera()
{
    Position.x=0.0;
    Position.y=0.0;
    Position.z=0.0;

    ViewDir.x=0.0;
    ViewDir.y=0.0;
    ViewDir.z=-1.0;

    RightVector.x=1.0;
    RightVector.y=0.0;
    RightVector.z=0.0;

    UpVector.x=0.0;
    UpVector.y=1.0;
    UpVector.z=0.0;

    target.x=0.0;
    target.y=0.0;
    target.z=-10.0;

    RotatedX = RotatedY = RotatedZ = 0.0;
}

void CCamera::Move (SF3dVector Direction)
{
    Position = Position + Direction;
}

void CCamera::RotateX (GLfloat Angle)
{
    Angle /= 10.0;
    RotatedX += Angle;

    ViewDir = ViewDir*cos(Angle*PIdiv180) + UpVector*sin(Angle*PIdiv180);

    ViewDir = Normalize3dVector(ViewDir);

    //now compute the new UpVector (by cross product)
    UpVector = CrossProduct(&ViewDir, &RightVector)*-1;
}

void CCamera::RotateY (GLfloat Angle)
{
    Angle /= 10.0;
    RotatedY += Angle;

    ViewDir = ViewDir*cos(Angle*PIdiv180) + RightVector*sin(Angle*PIdiv180);

    ViewDir = Normalize3dVector(ViewDir);

    //now compute the new RightVector (by cross product)
    RightVector = CrossProduct(&ViewDir, &UpVector);
}

void CCamera::RotateZ (GLfloat Angle)
{
    RotatedZ += Angle;

    //Rotate viewdir around the right vector:
    RightVector = Normalize3dVector(RightVector*cos(Angle*PIdiv180)
                                    + UpVector*sin(Angle*PIdiv180));

    //now compute the new UpVector (by cross product)
    UpVector = CrossProduct(&ViewDir, &RightVector)*-1;
}

void CCamera::RotateObjectX (GLfloat Angle)
{
    Angle /= 10.0;
    RotatedX += Angle;

    SF3dVector tmp;
    tmp = target - Position;

    SF3dVector tangeante;
    tangeante = CrossProduct(&tmp, &RightVector);
    tangeante = Normalize3dVector(tangeante);

    SF3dVector center_dir;
    center_dir = CrossProduct(&RightVector, &tangeante);
    center_dir = Normalize3dVector(center_dir);

    float dist = center_dir * tmp;

    Position = Position + center_dir*(1-cos(Angle*PIdiv180))*dist
            + tangeante*sin(Angle*PIdiv180)*dist;

    ViewDir = ViewDir*cos(Angle*PIdiv180) + UpVector*sin(Angle*PIdiv180);
    ViewDir = Normalize3dVector(ViewDir);

    //now compute the new UpVector (by cross product)
    UpVector = CrossProduct(&ViewDir, &RightVector)*-1;
}

void CCamera::RotateObjectY (GLfloat Angle)
{
    Angle /= 10.0;
    RotatedY += Angle;

    SF3dVector tmp;
    tmp = target - Position;

    SF3dVector tangeante;
    tangeante = CrossProduct(&tmp, &UpVector);
    tangeante = Normalize3dVector(tangeante);

    SF3dVector center_dir;
    center_dir = CrossProduct(&UpVector, &tangeante);
    center_dir = Normalize3dVector(center_dir);

    float dist = center_dir * tmp;

    Position = Position + center_dir*(1-cos(Angle*PIdiv180))*dist
            + tangeante*sin(Angle*PIdiv180)*dist;

    ViewDir = ViewDir*cos(Angle*PIdiv180) - RightVector*sin(Angle*PIdiv180);
    ViewDir = Normalize3dVector(ViewDir);

    //now compute the new RightVector (by cross product)
    RightVector = CrossProduct(&ViewDir, &UpVector);
}

void CCamera::Render( void )
{
    //The point at which the camera looks:
    SF3dVector ViewPoint = Position+ViewDir;

    //as we know the up vector, we can easily use gluLookAt:
    gluLookAt(	Position.x,Position.y,Position.z,
                ViewPoint.x,ViewPoint.y,ViewPoint.z,
                UpVector.x,UpVector.y,UpVector.z);
}

void CCamera::MoveForward( GLfloat Distance )
{
    Position = Position + (ViewDir*-Distance);
}

void CCamera::StrafeRight ( GLfloat Distance )
{
    Position = Position + (RightVector*Distance);
}

void CCamera::MoveUpward( GLfloat Distance )
{
    Position = Position + (UpVector*Distance);
}


void CCamera:: keyPressEvent(QKeyEvent* event)
{
    qDebug() << "Debug Message";
    if(event->key() == Qt::Key_Right)
    {
        SF3dVector tmp;
        tmp.x=0.1;
        tmp.y=0.0;
        tmp.z=0.0;

        Move(tmp);
        qDebug() << "Debug Message";
    }

}
