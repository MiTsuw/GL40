#include "ctrlwidget.h"

CtrlWidget::CtrlWidget(QWidget *parent, char* cfgFile) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->paramFrame->setWidgetsLink(ui->paintingMesh);
    param = new ConfigParams(cfgFile);
    string basename;
    param->readConfigParameter("input","inputSourceImageLeft", basename);
    NN inputNN;
    ImageRW<Point2D, GLfloat> input_read;
    input_read.read(basename, inputNN);
    NN resultNN;
    param->readConfigParameter("param_1","fileGrid2DPoints", basename);
    resultNN.read(basename);
    NNP3D ioNN, oNN;
    convert.doConversions(inputNN, resultNN, ioNN, oNN);
    oNN.write("output3D");
    ui->paintingMesh->initialize(param);
}

CtrlWidget::~CtrlWidget()
{
    delete ui;
}
