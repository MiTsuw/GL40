#ifndef PARAMFRAME_H
#define PARAMFRAME_H
//***************************************************************************
//
// Jean-Charles CREPUT, Abdelkhalek MANSOURI
// Created in 2013, Modified in 2015
//
//***************************************************************************

#include <QWidget>
#include "paintingmesh.h"
#include <QGroupBox>
#include <QHeaderView>
#include <QRadioButton>
#include <QApplication>
#include <QLabel>
#include <QHBoxLayout>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QRadioButton;
QT_END_NAMESPACE

class ParamFrame : public QWidget
{
    Q_OBJECT

private:
    QGroupBox *twoSidedGroupBox;
    QGroupBox *colorsGroupBox;
    QHBoxLayout *horizontalLayout;

    QRadioButton *view2DEnabledRadio;
    QRadioButton *view2DDisabledRadio;
    QRadioButton *colorsEnabledRadio;
    QRadioButton *colorsDisabledRadio;

    QGroupBox *displayGroupBox;
    QRadioButton *displayMRadio;
    QRadioButton *displayTRadio;
    QRadioButton *displayPRadio;
    QRadioButton *displayLRadio;
    QLabel *label;
    QVBoxLayout *verticalLayout;
    QVBoxLayout *verticalLayout_2;
    QGridLayout *gridLayout_2;

    PaintingMesh *pme;
public:
    explicit ParamFrame(QFrame *parent = 0)
    {

        horizontalLayout = new QHBoxLayout(parent);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString("horizontalLayout"));

        label = new QLabel(parent);
        label->setObjectName(QString("label"));
        label->setMaximumSize(QSize(250, 79));

        QFont font;
        font.setPointSize(15);
        font.setItalic(true);
        label->setFont(font);
        label->raise();

        horizontalLayout->addWidget(label);

        twoSidedGroupBox = new QGroupBox(parent);
        twoSidedGroupBox->setObjectName(QString("groupBox"));
        twoSidedGroupBox->setMaximumSize(QSize(180, 80));
        horizontalLayout->addWidget(twoSidedGroupBox);

        verticalLayout = new QVBoxLayout(twoSidedGroupBox);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString("verticalLayout"));


        view2DEnabledRadio = new QRadioButton(twoSidedGroupBox);
        view2DEnabledRadio->setObjectName(QString("radioButton"));

        view2DDisabledRadio = new QRadioButton(twoSidedGroupBox);
        view2DDisabledRadio->setObjectName(QString("radioButton_2"));

        view2DEnabledRadio->setChecked(true);
        verticalLayout->addWidget(view2DEnabledRadio);
        verticalLayout->addWidget(view2DDisabledRadio);

        colorsGroupBox = new QGroupBox(parent);
        colorsGroupBox->setObjectName(QString("groupBox_2"));
        colorsGroupBox->setMaximumSize(QSize(180, 80));

        horizontalLayout->addWidget(colorsGroupBox);

        verticalLayout_2 = new QVBoxLayout(colorsGroupBox);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString("verticalLayout_2"));

        colorsEnabledRadio = new QRadioButton(colorsGroupBox);
        colorsEnabledRadio->setObjectName(QString("radioButton_3"));
        colorsEnabledRadio->setGeometry(QRect(30, 30, 117, 22));

        colorsDisabledRadio = new QRadioButton(colorsGroupBox);
        colorsDisabledRadio->setObjectName(QString("radioButton_4"));
        colorsDisabledRadio->setGeometry(QRect(30, 60, 117, 22));

        colorsEnabledRadio->setChecked(true);
        verticalLayout_2->addWidget(colorsEnabledRadio);
        verticalLayout_2->addWidget(colorsDisabledRadio);

        displayGroupBox = new QGroupBox(parent);
        displayGroupBox->setObjectName(QString("groupBox_3"));
        displayGroupBox->setMinimumSize(QSize(310, 80));
        displayGroupBox->setMaximumSize(QSize(310, 80));
        horizontalLayout->addWidget(displayGroupBox);

        gridLayout_2 = new QGridLayout(displayGroupBox);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString("gridLayout_2"));

        displayMRadio = new QRadioButton(displayGroupBox);
        displayMRadio->setObjectName(QString("radioButton_5"));
        displayMRadio->setGeometry(QRect(30, 30, 117, 22));

        displayTRadio = new QRadioButton(displayGroupBox);
        displayTRadio->setObjectName(QString("radioButton_6"));
        displayTRadio->setGeometry(QRect(30, 60, 117, 22));

        displayPRadio = new QRadioButton(displayGroupBox);
        displayPRadio->setObjectName(QString("radioButton_7"));
        displayPRadio->setGeometry(QRect(230, 30, 117, 22));

        displayLRadio = new QRadioButton(displayGroupBox);
        displayLRadio->setObjectName(QString("radioButton_8"));
        displayLRadio->setGeometry(QRect(230, 60, 117, 22));

        displayMRadio->setChecked(true);
        gridLayout_2->addWidget(displayMRadio, 0, 0, 1, 1);
        gridLayout_2->addWidget(displayTRadio, 0, 1, 1, 1);
        gridLayout_2->addWidget(displayPRadio, 1, 0, 1, 1);
        gridLayout_2->addWidget(displayLRadio, 1, 1, 1, 1);

        twoSidedGroupBox->setTitle(QApplication::translate("MainWindow", "2D-3D", 0));
        view2DEnabledRadio->setText(QApplication::translate("MainWindow", "2D", 0));
        view2DDisabledRadio->setText(QApplication::translate("MainWindow", "3D", 0));
        colorsGroupBox->setTitle(QApplication::translate("MainWindow", "Colors", 0));
        colorsEnabledRadio->setText(QApplication::translate("MainWindow", "Enabled", 0));
        colorsDisabledRadio->setText(QApplication::translate("MainWindow", "Disabled", 0));
        displayGroupBox->setTitle(QApplication::translate("MainWindow", "Display", 0));
        displayMRadio->setText(QApplication::translate("MainWindow", "Mesh", 0));
        displayTRadio->setText(QApplication::translate("MainWindow", "Triangles", 0));
        displayPRadio->setText(QApplication::translate("MainWindow", "Points", 0));
        displayLRadio->setText(QApplication::translate("MainWindow", "Lines", 0));
        label->setText(QApplication::translate("MainWindow", "Display parameters", 0));


        connect(view2DEnabledRadio, SIGNAL(clicked()), this, SLOT(updateView()));
        connect(view2DDisabledRadio, SIGNAL(clicked()), this, SLOT(updateView()));
        connect(colorsEnabledRadio, SIGNAL(clicked()), this, SLOT(updateView()));
        connect(colorsDisabledRadio, SIGNAL(clicked()), this, SLOT(updateView()));

        connect(displayMRadio, SIGNAL(clicked()), this, SLOT(updateView()));
        connect(displayTRadio, SIGNAL(clicked()), this, SLOT(updateView()));
        connect(displayPRadio, SIGNAL(clicked()), this, SLOT(updateView()));
        connect(displayLRadio, SIGNAL(clicked()), this, SLOT(updateView()));



    }
    ~ParamFrame() {}

    void setWidgetsLink( PaintingMesh *pme) {
        this->pme = pme;
    }

private slots:

    void updateView()
    {
        if (view2DEnabledRadio->isChecked()) {
        } else if (view2DDisabledRadio->isChecked()) {
        }

        if (colorsEnabledRadio->isChecked()) {
            pme->modeColors = true;
            pme->makeObject();
            std::cout << "Colors" << endl;
        } else if (colorsDisabledRadio->isChecked()) {
            pme->modeColors = false;
            pme->makeObject();
            std::cout << "Colors disable" << endl;
        }
        if (displayMRadio->isChecked()) {
            pme->modeDisplay = 0;
        } else if (displayTRadio->isChecked()) {
            pme->modeDisplay = 1;
        } else if (displayPRadio->isChecked()) {
            pme->modeDisplay = 2;
        } else if (displayLRadio->isChecked()) {
            pme->modeDisplay = 3;
        }
    }

};

#endif // PARAMFRAME_H
