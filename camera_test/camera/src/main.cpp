#include <iostream>
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NIter.h"
#include "ViewGrid.h"

#include "TestCellular.h"
//#include "TestSom.h"

using namespace std;
using namespace components;
//using namespace operators;
using namespace meshing;

#define TEST_CODE  0
#define SECTION_PARAMETRES  0

int main(int argc, char *argv[])
{
    char* fileData;
    char* fileSolution;
    char* fileStats;
    char* fileConfig;

    /*
     * Lecture des fichiers d'entree
     */
    if (argc <= 1)
    {
        fileData = "input.data";
        fileSolution = "output.data";
        fileStats = "output.stats";
        fileConfig = "config.cfg";
    }
    else
    if (argc == 2)
    {
        fileData = argv[1];
        fileSolution = "output.data";
        fileStats = "output.stats";
        fileConfig = "config.cfg";
    }
    else
    if (argc == 3)
    {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = "output.stats";
        fileConfig = "config.cfg";
    }
    else
    if (argc == 4)
    {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = "config.cfg";
    }
    else
    if (argc >= 5)
    {
        fileData = argv[1];
        fileSolution = argv[2];
        fileStats = argv[3];
        fileConfig = argv[4];
    }
    //cout << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
    ConfigParams params(fileConfig);
    params.readConfigParameters();

    /*
     * Modification eventuelle des parametres
     */
#if! SECTION_PARAMETRES

    //[global_param]

    //# choix du mode de fonctionnement 0:evaluation, 1:som
    params.functionModeChoice = 2;

#endif //SECTION_PARAMETRES

    if (params.functionModeChoice == 0) {
        cout << "EVALUATE" << endl;
    }
    else if (params.functionModeChoice == 1) {
        cout << "================ Test for ViewGrid.h ====================" << endl;
        const size_t _X = 434, _Y = 383;
        const int R = 24;
        Grid<Point2D> carte(_X, _Y);
        for (int y = 0; y < _Y; y++)
        {
            for (int x = 0; x < _X; x++)
            {
                carte[y][x].set(0, x);
                carte[y][x].set(1, y);
            }
        }
        PointCoord initPoint(carte.getWidth() / 2, carte.getHeight() / 2 + 1);

        //TestViewGrid <ViewGridTetra, Grid<Point2D>, _X, _Y> t1(carte, initPoint, fileSolution);
       // t1.run();
        cout << "Fin de test " << params.functionModeChoice << '\n';
    }
    else if (params.functionModeChoice == 2) {
        cout << "TEST CELLULAR MATRIX" << endl;
        TestCellular t(fileData, fileSolution, fileStats, params);
        t.initialize();
        t.run();
        cout << "Fin de test " << params.functionModeChoice << '\n';
    }

    return 0;
}//main

