#include "paramframe.h"

ParamFrame::ParamFrame(QFrame *parent)
{

    btnStartZoom = new QPushButton("Start zoom");
    btnStopZoom= new QPushButton("Stop zoom");
    tZoomCamera=new MyThread(this);

    verticalLayout_0 = new QVBoxLayout(parent);
    verticalLayout_0->setSpacing(6);
    verticalLayout_0->setContentsMargins(11, 11, 11, 11);
    verticalLayout_0->setObjectName(QString("verticalLayout_0"));

    label = new QLabel(parent);
    label->setObjectName(QString("label"));
    label->setMaximumSize(QSize(250, 79));

    QFont font;
    font.setPointSize(15);
    font.setItalic(true);
    label->setFont(font);
    label->raise();

    verticalLayout_0->addWidget(label);

    verticalLayout_0->addWidget(btnStartZoom);
    verticalLayout_0->addWidget(btnStopZoom);

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



    connect(tZoomCamera, SIGNAL(updateScreen()), this, SLOT(autoSelfZoom()));
    connect(btnStartZoom, SIGNAL(clicked()), this, SLOT(startZoom()));
    connect(btnStopZoom, SIGNAL(clicked()), this, SLOT(stopZoom()));

    connect(view2DEnabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));
    connect(view2DDisabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));
    connect(colorsEnabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));
    connect(colorsDisabledButton, SIGNAL(clicked(bool)), this, SLOT(updateView()));

    connect(displayMButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));
    connect(displayTButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));
    connect(displayPButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));
    connect(displayLButton, SIGNAL(clicked()), this, SLOT(updateDisplay()));



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



void ParamFrame::autoSelfZoom()
{
    pme->selfZoom();
}

// actions start et pause


void ParamFrame:: startZoom() {
    tZoomCamera->Stop = false;
    tZoomCamera->start();
}

void ParamFrame:: stopZoom() {
    tZoomCamera->Stop = true;
}

