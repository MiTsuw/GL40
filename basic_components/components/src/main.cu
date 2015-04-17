#include <iostream>
#include "ConfigParams.h"
#include "Node.h"
#include "GridOfNodes.h"
//#include "binary_operations.h"
#include "NeighborhoodIterator.h"
#include "ViewGrid.h"

using namespace std;
using namespace components;

#define TEST_CODE  0
#define SECTION_PARAMETRES  1

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
//    cout << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
//    ConfigParams params(fileConfig);
//    params.readConfigParameters();

    /*
     * Modification eventuelle des parametres
     */
#if SECTION_PARAMETRES

    //[global_param]

    //# choix du mode de fonctionnement 0:evaluation, 1:local search,
    //# 2:genetic algorithm
    int functionModeChoice = 1;

#endif //SECTION_PARAMETRES
    cout << "Test " << functionModeChoice << '\n';
    if (functionModeChoice == 0) {
//        Test<GLfloat, 21, 21, 0, 10, NIterHexa > t0(0);
//        t0.run();
//        Test<GLfloat, 21, 21, 6, 7, NIterHexa > t1(0);
//        t1.run();
//        Test<GLfloat, 21, 21, 0, 9, NIterTetra > t2(0);
//        t2.run();
//        Test<GLfloat, 21, 21, 6, 7, NIterTetra > t3(0);
//        t3.run();
        Test<typename Point2D, 230, 190, 0, 100, NIterHexa > t4(Point2D(5,5));
        t4.run();
        cout << "Fin de test " << functionModeChoice << '\n';
    }
    else if (functionModeChoice == 1) {
        cout << "================ Test for ViewGrid.h ====================" << endl;
        const size_t _X = 434, _Y = 383;
        const int R = 23;
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

        TestViewGrid <ViewGridHexa<R>, Grid<Point2D>, _X, _Y> t1(carte, initPoint, fileSolution);
        t1.run();
        cout << "Fin de test " << functionModeChoice << '\n';
    }
#if TEST_CODE
    else if (params.functionModeChoice == 2) {
        Test<GLfloat, 10, 15, Addition<GLfloat> > t(2, 3, -5);
        t.run();
        Test<Point2D, 10, 15, Difference<Point2D> > t2(Point2D(2,2), Point2D(3,3), Point2D(-5,-5));
        t2.run();
        Test<Point3D, 10, 15, Mult<Point3D> > t3(Point3D(2,2,2), Point3D(3,3,3), Point3D(-5,-5,-5));
        t3.run();

        // Create density map
        GridDensity dm;
        dm.resize(10,10);

        dm.reset_value(0);

        dm.set(5, 5, 10);
        dm.set(4, 5, 9);

        // Create grid points
        Grid2DPoints gd;
        gd.resize(10,10);

        gd.reset_value(Point2D(0,0));

        Point2D p(8, 8);
        gd.set(5, 5, p);

        p.set<0>(3);
        p.set<1>(2);
        gd.set(4, 5, p);

        // Writting
        ofstream fo;
        fo.open(fileData);
        if (fo) {
            fo << dm;
            fo.close();
        }

        // Writting
        fo.open(fileSolution);
        if (fo) {
            fo << gd;
            fo.close();
        }
    }
    else    if (params.functionModeChoice == 3) {

        // For Reading
        ifstream fi;

        // Read density map
        GridDensity dm;
        fi.open(fileData);
        if (!fi) {
            std::cout << "erreur ouverture GridDensity" << endl;
            exit(0);
        }
        else
            fi >> dm;
        fi.close();

        // Read grid of neurons
        Grid2DPoints gd;
        fi.open(fileSolution);
        if (!fi) {
            std::cout << "erreur ouverture Grid2DPoints" << endl;
            exit(0);
        }
        else
            fi >> gd;
        fi.close();

        dm.set(7, 5, 10);
        dm.set(6, 5, 9);

        Point2D p(9, 9);
        gd.set(6, 5, p);

        p.set<0>(8);
        p.set<1>(8);
        gd.set(7, 5, p);

        // Writting
        ofstream fo;
        fo.open(fileData);
        if (fo) {
            fo << dm;
            fo.close();
        }

        // Writting
        fo.open(fileSolution);
        if (fo) {
            fo << gd;
            fo.close();
        }
    }
    else    if (params.functionModeChoice == 4) {
        binary_operations::Test test;
        test.run();
        Objectives objs;

    }
#endif
    cout << "Exit program " << '\n';

    return 0;
}

