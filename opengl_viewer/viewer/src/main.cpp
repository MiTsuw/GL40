#include <QApplication>
#include <cstdlib>

#include "ctrlwidget.h"

char cfgFile[256];

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    if (argc <= 1)
    {
        cout << argv[0] << " configuration file : default config.cfg" << endl;
        strcpy(cfgFile, "config.cfg");
    }
    else
    if (argc == 2)
    {
        strcpy(cfgFile, argv[1]);
        cout << argv[0] << " configuration file : " << cfgFile << endl;
    }

    CtrlWidget w(0, cfgFile);

    w.show();

    return a.exec();
}
