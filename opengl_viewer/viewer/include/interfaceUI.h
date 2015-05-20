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

#include "paintingmesh.h"
#include "paramframe.h"
#include <QMainWindow>
#include "paramframe.h"


class Ui_MainWindow : public QWidget
{

    Q_OBJECT

protected:
    QTime m_timer;
    int m_frameCount;

public:
    QMenuBar* menuBar;
    QToolBar* toolBar;
    QStatusBar* statusBar;
    string curPix;          // Image actuellement visualisé
    QMenu* lpModeMenu;      // Menu Mode Leap Motion
    QMenu* aboutMenu;       // Menu A propos

    QAction * quitApp;
    QAction * zoomIn;
    QAction * zoomOut;
    QAction * resetCam;
    QAction * resetAll;

    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QFrame *frame;
    QFrame *frame_2;
    QWidget *Mesh;
    QGridLayout *gridLayout_5;
    PaintingMesh *paintingMesh;
    ParamFrame *paramFrame;



    void setupUi(QMainWindow *MainWindow)
    {
        int largeur = QApplication::desktop()->width();
        int hauteur = QApplication::desktop()->height();

        createMenus(MainWindow);
        createToolBar(MainWindow);
        createStatusBar(MainWindow);


        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString("MainWindow"));
            MainWindow->resize(largeur, hauteur);
            MainWindow->setAnimated(true);

            centralWidget = new QWidget(MainWindow);
            centralWidget->setObjectName(QString("centralWidget"));
            centralWidget->setEnabled(true);

            gridLayout = new QGridLayout(centralWidget);
            gridLayout->setSpacing(6);
            gridLayout->setContentsMargins(11, 11, 11, 11);
            gridLayout->setObjectName(QString("gridLayout"));

            frame = new QFrame(centralWidget);
            frame->setObjectName(QString("frame"));
            frame->setMaximumSize(QSize(16777215, 101));
            frame->setSizeIncrement(QSize(0, 0));
            frame->setFrameShape(QFrame::StyledPanel);
            frame->setFrameShadow(QFrame::Raised);

            gridLayout->addWidget(frame, 0, 0, 1, 2);

            paramFrame = new ParamFrame(frame);

            frame_2 = new QFrame(centralWidget);
            frame_2->setObjectName(QString("frame_2"));
            frame_2->setMaximumSize(QSize(120, 16777215));
            frame_2->setFrameShape(QFrame::StyledPanel);
            frame_2->setFrameShadow(QFrame::Raised);

            gridLayout->addWidget(frame_2, 1, 0, 1, 1);

            Mesh = new QWidget(centralWidget);
            Mesh->setObjectName(QString("Mesh"));
            Mesh->setEnabled(true);

            gridLayout_5 = new QGridLayout(Mesh);
            gridLayout_5->setSpacing(6);
            gridLayout_5->setContentsMargins(11, 11, 11, 11);
            gridLayout_5->setObjectName(QString("gridLayout_5"));

            paintingMesh = new PaintingMesh(Mesh);
            paintingMesh->setObjectName(QString("paintingMesh"));

            gridLayout_5->addWidget(paintingMesh);

            gridLayout->addWidget(Mesh, 1, 1, 1, 1);
        MainWindow->setCentralWidget(centralWidget);
        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    }

    /* Création des Menus */
    void createMenus(QMainWindow *MainWindow)
    {
        menuBar = new QMenuBar();                // On crée la barre de menu
        MainWindow->setMenuBar(menuBar);

          /* Menu Mode Leap Motion */
          lpModeMenu = menuBar->addMenu("&Mode Leap Motion");
          //lpModeMenu->setIcon(QIcon("icones/lpLogo.png"));
          lpModeMenu->setDisabled(true);

          menuBar->addSeparator();

          /* Menu A propos */
          aboutMenu = menuBar->addMenu("&A propos");
          //aboutMenu->setIcon(QIcon("icones/pointInterrogation.png"));
          QObject::connect(aboutMenu, SIGNAL(triggered(QAction *)), this, SLOT(apropos()));
    }

    /* Création des Menus */
    void createToolBar(QMainWindow *MainWindow)
    {
        toolBar = new QToolBar();
        MainWindow->addToolBar(toolBar);
        toolBar->setMovable(false);
        zoomIn = toolBar->addAction(/*QIcon(newpix),*/ "Zoom +");
        zoomOut = toolBar->addAction(/*QIcon(openpix),*/ "Zoom -");
        toolBar->addSeparator();
        resetCam = toolBar->addAction(/*QIcon(openpix),*/ "Reset camera");
        toolBar->addSeparator();
        resetAll = toolBar->addAction(/*QIcon(openpix),*/ "Reset all");
        toolBar->addSeparator();
        quitApp = toolBar->addAction(/*QIcon(quitpix),*/"Quit Application");

        //QObject::connect(zoomIn, SIGNAL(triggered()), this, SLOT(zoomInCamera()));
        //QObject::connect(zoomOut, SIGNAL(triggered()), this, SLOT(zoomOutCamera()));
        //QObject::connect(resetCam, SIGNAL(clicked()), this, SLOT(resetCamera()))
        QObject::connect(quitApp, SIGNAL(triggered()), this, SLOT(quitappp()));
   }

    /* Création de la barre de statut */
    void createStatusBar(QMainWindow *MainWindow)
    {
        statusBar = new QStatusBar();
        MainWindow->setStatusBar(statusBar);
        /* C'est sensé afficher les fps........... lel ca te mets une petite note de musique */
        statusBar->showMessage(QString("FPS: %f 72.0f " /*/m_frameCount/(float(m_timer.elapsed())/1000.0f)*/));
    }

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Viewer to infinity and beyond", 0));
    }
private slots:
    void quitappp()
    {
        exit(0);
    }

    void apropos()
    {
        QMessageBox msgBox;
         msgBox.setText("Le document a été modifié.");
         msgBox.exec();
    }

};

namespace Ui {
    class MainWindow: public Ui_MainWindow{};
}

#endif // INTERFACUI_H
