#ifndef INTERFACUI_H
#define INTERFACUI_H
//***************************************************************************
//
// Ahmet IMRE, Barbara SCHIAVI, Constantin JEAN, Victor GABRIEL
//
//***************************************************************************

#include <QDesktopWidget>
#include <QtCore/QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QRadioButton>
#include <QGroupBox>
#include <QDialog>
#include <QToolBar>
#include <QMenuBar>
#include <QStatusBar>
#include <QMenu>
#include <QGridLayout>
#include <QHeaderView>
#include <QPushButton>
#include <QTabWidget>
#include <QWidget>
#include <QMessageBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QRect>
#include <QPushButton>
#include "paintingmesh.h"
#include "paramframe.h"
#include <QMainWindow>
#include "camera.h"
#include <QGLWidget>


class Ui_MainWindow : public QWidget
{

    Q_OBJECT
private:
    QMenuBar* menuBar;
    QToolBar* toolBar;
    QStatusBar* statusBar;
    string curPix;          // Image actuellement visualisé
    QMenu* lpModeMenu;      // Menu Mode Leap Motion
    QMenu* aboutMenu;       // Menu A propos
    QMenu *m_file;
    QMenu *m_edit;
    QMenu *m_display;
    QMenu *m_help;


    /*QAction * quitApp;
    QAction * zoomIn;
    QAction * zoomOut;
    QAction * resetCam;
    QAction * resetAll;*/

    QAction *a_close;
    QAction *a_open;
    QAction *a_save;
    QAction *a_cameraReset;
    QAction *a_about;
    QAction *a_sclist;




    QPushButton * myButton;

    CCamera camera;

    QLabel * lblcoordX;
    QLabel * lblcoordY;
    QLabel * lblzoom;
    QLabel * lblLeapMotion;

    QLineEdit * lecoordX;
    QLineEdit * lecoordY;
    QLineEdit * lezoom;

    QCheckBox * cbLeapMotion;

    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QFrame *frame;
    QFrame *frame_2;
    QWidget *Mesh;
    QGridLayout *gridLayout_5;


protected:
    QTime m_timer;
    int m_frameCount;

public:

    void setupUi(QMainWindow *);
    void createMenus(QMainWindow *);
    void createToolBar(QMainWindow *);
    void createStatusBar(QMainWindow *);
    void retranslateUi(QMainWindow *);
    PaintingMesh *paintingMesh;
    ParamFrame *paramFrame;

public slots:
    void quitappp();
    void apropos();
    void open();
    void save();
    void close();
    void cameraReset();

};

namespace Ui {
class MainWindow: public Ui_MainWindow{};
}

#endif // INTERFACUI_H
