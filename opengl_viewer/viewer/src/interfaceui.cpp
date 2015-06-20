#include "interfaceUI.h"


    void Ui_MainWindow::setupUi(QMainWindow *MainWindow)
    {
        /*int largeur = QApplication::desktop()->width();
        int hauteur = QApplication::desktop()->height();
        */
        createMenus(MainWindow);
        createToolBar(MainWindow);
        createStatusBar(MainWindow);

        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString("MainWindow"));
        MainWindow->resize(1280, 680);
        MainWindow->setAnimated(true);

        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString("centralWidget"));
        centralWidget->setEnabled(true);

        gridLayout = new QGridLayout(centralWidget);
        //gridLayout->setSpacing(6);
        //gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString("gridLayout"));

        frame = new QFrame(centralWidget);
        frame->setObjectName(QString("frame"));
        //frame->setMaximumSize(QSize(16777215, 101));
        //frame->setSizeIncrement(QSize(0, 0));
        //frame->setFrameShape(QFrame::StyledPanel);
        //frame->setFrameShadow(QFrame::Raised);

        gridLayout->addWidget(frame, 0, 0,1,1);

        paramFrame = new ParamFrame(frame);
        frame_2 = new QFrame();
        frame_2->setObjectName(QString("frame_2"));
        frame_2->setMaximumSize(QSize(1024, 640));
        frame_2->setMinimumSize(QSize(1024,480 ));
        //frame_2->setFrameShape(QFrame::StyledPanel);
        //frame_2->setFrameShadow(QFrame::Raised);

        //gridLayout->addWidget(frame_2,0,1);

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

        gridLayout->addWidget(Mesh, 0,1,1,4);
        MainWindow->setCentralWidget(centralWidget);
        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    }

    /* Création des Menus */
    void Ui_MainWindow::createMenus(QMainWindow *MainWindow)
    {
        menuBar = new QMenuBar();                // On crée la barre de menu
        MainWindow->setMenuBar(menuBar);

        //création des menus
        //QList<QMenu*> menuList;

        // ******************** FILE MENU ******************
        m_file = new QMenu("File");

        a_open = new QAction("&Open",MainWindow);
        //a_open.setText("Open File");
        a_open->setShortcut(QKeySequence::Open);
        connect(a_open, SIGNAL(triggered()), this, SLOT(open()));

        a_save = new QAction("&Save",MainWindow);
        a_save->setShortcut(QKeySequence::Save);
        connect(a_save, SIGNAL(triggered()), this, SLOT(save()));

        a_close = new QAction("&Close",MainWindow);
        a_close->setShortcut(QKeySequence::Close);
        connect(a_close, SIGNAL(triggered()), this, SLOT(close()));

        m_file->addAction(a_open);
        m_file->addAction(a_save);
        m_file->addAction(a_close);

        // ******************** EDIT MENU *****************
        m_edit = new QMenu("Edit");



        // ********************* DISPLAY MENU ***************
        m_display = new QMenu("Display");
        a_cameraReset = new QAction("&Reset Camera",MainWindow);
        m_display->addAction(a_cameraReset);
        connect(a_cameraReset, SIGNAL(triggered()), this, SLOT(cameraReset()));

        // ********************* HELP MENU ******************
        m_help = new QMenu("Help");
        a_about = new QAction("&About",MainWindow);
        a_sclist = new QAction("&Shortcut List",MainWindow);
        m_help->addAction(a_about);
        m_help->addAction(a_sclist);

        menuBar->addSeparator();

        // Ajout des menus à la menubar
        /*for(i=0;i<menuList.size();i++)
        menuBar->addMenu(menuList(i));*/
        menuBar->addMenu(m_file);
        menuBar->addMenu(m_edit);
        menuBar->addMenu(m_display);
        menuBar->addMenu(m_help);

    }

    /* Création du toolbar */
    void Ui_MainWindow::createToolBar(QMainWindow *MainWindow)
    {
        toolBar = new QToolBar();
        MainWindow->addToolBar(toolBar);
        toolBar->setMovable(false);
        lblcoordX = new QLabel();
        lblcoordY = new QLabel();
        lblzoom = new QLabel();
        lblLeapMotion = new QLabel();
        lecoordX = new QLineEdit();
        lecoordY = new QLineEdit();
        lezoom = new QLineEdit();
        cbLeapMotion = new QCheckBox();


        lblcoordX->setText("     X :");
        lblcoordY->setText("     Y :");
        lblzoom->setText("     Zoom :");
        lblLeapMotion->setText(" Toggle Leap Motion Mod ");
        lecoordX->setText("0,0 px");
        lecoordX->setMaximumWidth(80);
        lecoordY->setText("0,0 px");
        lecoordY->setMaximumWidth(80);
        lezoom->setText("100 %");
        lezoom->setMaximumWidth(80);

        toolBar->addWidget(lblcoordX);
        toolBar->addWidget(lecoordX);
        toolBar->addWidget(lblcoordY);
        toolBar->addWidget(lecoordY);
        toolBar->addWidget(lblzoom);
        toolBar->addWidget(lezoom);
        toolBar->addSeparator();
        toolBar->addWidget(cbLeapMotion);
        toolBar->addWidget(lblLeapMotion);

    }

    /* Création de la barre de statut */
    void Ui_MainWindow::createStatusBar(QMainWindow *MainWindow)
    {
        statusBar = new QStatusBar();
        MainWindow->setStatusBar(statusBar);
        statusBar->showMessage(QString("FPS: 72"/*/m_frameCount/(float(m_timer.elapsed())/1000.0f)*/));
    }

    void Ui_MainWindow::retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Viewer to infinity and beyond", 0));
    }

    void Ui_MainWindow::quitappp()
    {
        exit(0);
    }

    void Ui_MainWindow::apropos()
    {
        QMessageBox msgBox;
        msgBox.setText("Le document a été modifié.");
        msgBox.exec();
    }

    void Ui_MainWindow::open()
    {

    }

    void Ui_MainWindow::save()
    {

    }

    void Ui_MainWindow::zoomauto()
    {
        /*MyThread mt = new MyThread(paintingMesh, 1);

        this->moveToThread(mt);
        mt->start();*/

    }

    void Ui_MainWindow::close()
    {
        cout<<"jeej"<<endl;
    }

    void Ui_MainWindow::cameraReset()
    {
        paintingMesh->reinitCamera();
    }

