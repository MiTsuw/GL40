#include "paramframe.h"

ParamFrame::ParamFrame(QFrame *parent)
{
    verticalLayout_0 = new QVBoxLayout(parent);
    verticalLayout_0->setSpacing(6);
    verticalLayout_0->setContentsMargins(11, 11, 11, 11);
    verticalLayout_0->setObjectName(QString("verticalLayout_0"));





    //Création des threads zoom et rotation
    tZoomCamera=new MyThread(this,0);
    tRotateCamera=new MyThread(this,1);


    label = new QLabel(parent);
    label->setObjectName(QString("label"));
    label->setMaximumSize(QSize(250, 79));

    QFont font;
    font.setPointSize(15);
    font.setItalic(true);
    label->setFont(font);
    label->raise();

    verticalLayout_0->addWidget(label);


/*/////////////////////////////////////////////////////////////////////////*/
//  Ajouts des boutons pour le zoom
/*/////////////////////////////////////////////////////////////////////////*/

    //Création du group box pour les boutons de zoom
    zoomGroupBox = new QGroupBox(parent);
    zoomGroupBox->setObjectName(QString("groupBoxZoom"));
    zoomGroupBox->setMaximumSize(QSize(180, 80));
    verticalLayout_0->addWidget(zoomGroupBox);


    //Création du layout
    horizontalLayout_3 = new QHBoxLayout(zoomGroupBox);
    horizontalLayout_3->setSpacing(0);
    horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
    horizontalLayout_3->setObjectName(QString("horizontalLayout_3"));

    //creation des Icones
    playIcon = new QIcon(":/icons/play.png");
    stopIcon = new QIcon(":/icons/stop.png");

    //Création des boutons de zoom
    btnStartZoom = new QPushButton(*playIcon,"", zoomGroupBox);
    btnStartZoom->setIconSize(QSize(20,20));

    btnStopZoom= new QPushButton(*stopIcon,"",zoomGroupBox);
    btnStopZoom->setIconSize(QSize(20,20));

    sliderZoom=new QSlider(/*Qt::Horizontal*/);
    sliderZoom->setMinimum(-9);
    sliderZoom->setMaximum(9);
    sliderZoom->setValue(1);
    //sliderZoom->setVisible(false);

    horizontalLayout_3->addWidget(btnStartZoom);
    horizontalLayout_3->addWidget(btnStopZoom);
    horizontalLayout_3->addWidget(sliderZoom);

/*/////////////////////////////////////////////////////////////////////////*/
//  Ajouts des boutons pour la rotation de la camera
/*/////////////////////////////////////////////////////////////////////////*/

    //Création du group box pour les boutons de rotations
    rotationGroupBox = new QGroupBox(parent);
    rotationGroupBox->setObjectName(QString("groupBoxRotation"));
    rotationGroupBox->setMaximumSize(QSize(180, 80));
    verticalLayout_0->addWidget(rotationGroupBox);


    //Création du layout
    horizontalLayout_4 = new QHBoxLayout(rotationGroupBox);
    horizontalLayout_4->setSpacing(0);
    horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
    horizontalLayout_4->setObjectName(QString("horizontalLayout_4"));


    //Création des boutons et du thread zoom
    btnStartRotation = new QPushButton(*playIcon,"", rotationGroupBox);
    btnStartRotation->setIconSize(QSize(20,20));

    btnStopRotation= new QPushButton(*stopIcon,"",rotationGroupBox);
    btnStopRotation->setIconSize(QSize(20,20));

    sliderRotation=new QSlider(/*Qt::Horizontal*/);
    sliderRotation->setValue(0);
    sliderRotation->setMinimum(0);
    sliderRotation->setMaximum(1);

    horizontalLayout_4->addWidget(btnStartRotation);
    horizontalLayout_4->addWidget(btnStopRotation);
    horizontalLayout_4->addWidget(sliderRotation);


/*/////////////////////////////////////////////////////////////////////////*/
//  Ajouts des boutons 2D-3D
/*/////////////////////////////////////////////////////////////////////////*/

    twoSidedGroupBox = new QGroupBox(parent);
    twoSidedGroupBox->setObjectName(QString("groupBox"));
    twoSidedGroupBox->setMaximumSize(QSize(180, 80));
    verticalLayout_0->addWidget(twoSidedGroupBox);

    horizontalLayout_1 = new QHBoxLayout(twoSidedGroupBox);
    horizontalLayout_1->setSpacing(0);
    horizontalLayout_1->setContentsMargins(11, 11, 11, 11);
    horizontalLayout_1->setObjectName(QString("horizontalLayout_1"));


    view2DEnabledButton = new QPushButton(twoSidedGroupBox);
    view2DEnabledButton->setObjectName(QString("pushButton"));
    view2DEnabledButton->setCheckable(true);
    view2DEnabledButton->setAutoExclusive(true);

    view2DDisabledButton = new QPushButton(twoSidedGroupBox);
    view2DDisabledButton->setObjectName(QString("pushButton_2"));
    view2DDisabledButton->setCheckable(true);
    view2DDisabledButton->setChecked(true);
    view2DDisabledButton->setAutoExclusive(true);

    //view2DEnabledButton->setChecked(true);
    horizontalLayout_1->addWidget(view2DEnabledButton);
    horizontalLayout_1->addWidget(view2DDisabledButton);


/*/////////////////////////////////////////////////////////////////////////*/
//  Ajouts des boutons pour la couleur
/*/////////////////////////////////////////////////////////////////////////*/
    colorsGroupBox = new QGroupBox(parent);
    colorsGroupBox->setObjectName(QString("groupBox_2"));
    colorsGroupBox->setMaximumSize(QSize(180, 80));

    verticalLayout_0->addWidget(colorsGroupBox);

    horizontalLayout_2 = new QHBoxLayout(colorsGroupBox);
    horizontalLayout_2->setSpacing(0);
    horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
    horizontalLayout_2->setObjectName(QString("horizontalLayout_2"));

    colorsEnabledButton = new QPushButton(colorsGroupBox);
    colorsEnabledButton->setObjectName(QString("pushButton_3"));
    colorsEnabledButton->setGeometry(QRect(30, 30, 117, 22));
    colorsEnabledButton->setCheckable(true);
    colorsEnabledButton->setAutoExclusive(true);

    colorsDisabledButton = new QPushButton(colorsGroupBox);
    colorsDisabledButton->setObjectName(QString("pushButton_4"));
    colorsDisabledButton->setGeometry(QRect(30, 60, 117, 22));
    colorsDisabledButton->setCheckable(true);
    colorsDisabledButton->setAutoExclusive(true);

    colorsEnabledButton->setChecked(true);
    horizontalLayout_2->addWidget(colorsEnabledButton);
    horizontalLayout_2->addWidget(colorsDisabledButton);



/*/////////////////////////////////////////////////////////////////////////*/
//  Ajouts des boutons display
/*/////////////////////////////////////////////////////////////////////////*/

    displayGroupBox = new QGroupBox(parent);
    displayGroupBox->setObjectName(QString("groupBox_3"));
    displayGroupBox->setMinimumSize(QSize(180, 150));
    displayGroupBox->setMaximumSize(QSize(180, 150));
    verticalLayout_0->addWidget(displayGroupBox);

    /*gridLayout_2 = new QGridLayout(displayGroupBox);
    gridLayout_2->setSpacing(6);
    gridLayout_2->setContentsMargins(11, 11, 11, 11);
    gridLayout_2->setObjectName(QString("gridLayout_2"));
*/
    displayMButton = new QPushButton(displayGroupBox);
    displayMButton->setObjectName(QString("pushButton_5"));
    displayMButton->setGeometry(QRect(30, 30, 117, 22));
    displayMButton->setCheckable(true);
    displayMButton->setAutoExclusive(true);

    displayTButton = new QPushButton(displayGroupBox);
    displayTButton->setObjectName(QString("radioButton_6"));
    displayTButton->setGeometry(QRect(30, 60, 117, 22));
    displayTButton->setCheckable(true);
    displayTButton->setAutoExclusive(true);

    displayPButton = new QPushButton(displayGroupBox);
    displayPButton->setObjectName(QString("radioButton_7"));
    displayPButton->setGeometry(QRect(30, 90, 117, 22));
    displayPButton->setCheckable(true);
    displayPButton->setAutoExclusive(true);

    displayLButton = new QPushButton(displayGroupBox);
    displayLButton->setObjectName(QString("radioButton_8"));
    displayLButton->setGeometry(QRect(30, 120, 117, 22));
    displayLButton->setCheckable(true);
    displayLButton->setAutoExclusive(true);

    displayMButton->setChecked(true);
    /*gridLayout_2->addWidget(displayMRadio, 0, 0, 1, 1);
    gridLayout_2->addWidget(displayTRadio, 1, 0, 1, 1);
    gridLayout_2->addWidget(displayPRadio, 2, 0, 1, 1);
    gridLayout_2->addWidget(displayLRadio, 3, 0, 1, 1);


*/

    btnRefresh=new QPushButton("Refresh");
    verticalLayout_0->addWidget(btnRefresh);

    rotationGroupBox->setTitle(QApplication::translate("MainWindow", "Rotation", 0));
    zoomGroupBox->setTitle(QApplication::translate("MainWindow", "Zoom", 0));
    twoSidedGroupBox->setTitle(QApplication::translate("MainWindow", "2D-3D", 0));
    view2DEnabledButton->setText(QApplication::translate("MainWindow", "2D", 0));
    view2DDisabledButton->setText(QApplication::translate("MainWindow", "3D", 0));
    colorsGroupBox->setTitle(QApplication::translate("MainWindow", "Colors", 0));
    colorsEnabledButton->setText(QApplication::translate("MainWindow", "Enabled", 0));
    colorsDisabledButton->setText(QApplication::translate("MainWindow", "Disabled", 0));
    displayGroupBox->setTitle(QApplication::translate("MainWindow", "Display", 0));
    displayMButton->setText(QApplication::translate("MainWindow", "Mesh", 0));
    displayTButton->setText(QApplication::translate("MainWindow", "Triangles", 0));
    displayPButton->setText(QApplication::translate("MainWindow", "Points", 0));
    displayLButton->setText(QApplication::translate("MainWindow", "Lines", 0));
    label->setText(QApplication::translate("MainWindow", "Display parameters", 0));


    //On connecte le thread zoom et les boutons qui lui sont liés
    connect(tZoomCamera, SIGNAL(updateScreen(int)), this, SLOT(autoSelfZoom(int)));
    connect(sliderZoom,SIGNAL(valueChanged(int)),tZoomCamera,SLOT(updateSpeed(int)));
    connect(btnStartZoom, SIGNAL(clicked()), this, SLOT(startZoom()));
    connect(btnStopZoom, SIGNAL(clicked()), this, SLOT(stopZoom()));


    //On connecte le thread rotate et les boutons qui lui sont liés
    connect(tRotateCamera, SIGNAL(updateScreen(int)), this, SLOT(autoSelfRotate(int)));
    connect(sliderRotation,SIGNAL(valueChanged(int)), tRotateCamera, SLOT(updateModeRotation(int)));
    connect(btnStartRotation, SIGNAL(clicked()), this, SLOT(startRotate()));
    connect(btnStopRotation, SIGNAL(clicked()), this, SLOT(stopRotate()));


    connect(view2DEnabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));
    connect(view2DDisabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));
    connect(colorsEnabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));
    connect(colorsDisabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));

    connect(displayMButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));
    connect(displayTButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));
    connect(displayPButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));
    connect(displayLButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));

    connect(btnRefresh, SIGNAL(clicked()), this, SLOT(refreshCamera()));



}
ParamFrame::~ParamFrame() {}

void ParamFrame::setWidgetsLink( PaintingMesh *pme) {
    this->pme = pme;
}


void ParamFrame::updateView(){
    if (view2DEnabledButton->isChecked()) {


    } else if (view2DDisabledButton->isChecked()) {

    }

    if (colorsEnabledButton->isChecked()) {
        pme->modeColors = true;
        pme->makeObject();
        std::cout << "Colors" << endl;
    } else if (colorsDisabledButton->isChecked()) {
        pme->modeColors = false;
        pme->makeObject();
        std::cout << "Colors disable" << endl;
    }

}

void ParamFrame::updateDisplay()
{
    if (displayMButton->isChecked()) {
        pme->modeDisplay = 0;
    } else if (displayTButton->isChecked()) {
        pme->modeDisplay = 1;
    } else if (displayPButton->isChecked()) {
        pme->modeDisplay = 2;
    } else if (displayLButton->isChecked()) {
        pme->modeDisplay = 3;
    }
}


//Thread zoom
void ParamFrame::autoSelfZoom(int v)
{
    //qDebug()<<"v"<<v<<endl;
    pme->selfZoom(v);
}

void ParamFrame:: startZoom() {
   // sliderZoom->setVisible(true);
    tZoomCamera->Stop = false;
    tZoomCamera->setType(0);
    tZoomCamera->start();
}

void ParamFrame:: stopZoom() {
   // sliderZoom->setVisible(false);
    tZoomCamera->Stop = true;
}

//Thread zoom
void ParamFrame::autoSelfRotate(int m)
{
    //qDebug()<<"m"<<m<<endl;
    pme->selfRotate(m);
}

void ParamFrame:: startRotate() {
    tRotateCamera->Stop = false;
    tRotateCamera->setType(1);
    tRotateCamera->start();
}

void ParamFrame:: stopRotate() {
    tRotateCamera->Stop = true;
}

void ParamFrame:: refreshCamera()
{
    pme->reinitCamera();

}


