#include <iostream>
#include "ConfigParams.h"
//#include "InstanceGenerator.h"
#include "Node.h"
#include "GridOfNodes.h"
#include "NeighborhoodIterator.h"
#include "CellularMatrix.h"
//#include "SomOperator.h"

using namespace std;
using namespace components;
using namespace operators;

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
    //cout << argv[0] << " " << fileData << " " << fileSolution << " " << fileStats << " " << fileConfig << endl;

    /*
     * Lecture des parametres
     */
    ConfigParams params(fileConfig);
    params.readConfigParameters();

    /*
     * Modification eventuelle des parametres
     */
#if SECTION_PARAMETRES

    //[global_param]

    //# choix du mode de fonctionnement 0:evaluation, 1:som
    params.functionModeChoice = 1;

#endif //SECTION_PARAMETRES

    if (params.functionModeChoice == 0) {
        cout << "EVALUATE" << endl;
    }
    else if (params.functionModeChoice == 1) {
        cout << "TEST CELLULAR MATRIX" << endl;
        TestCellular t;
        t.run();
        cout << "Fin de test " << params.functionModeChoice << '\n';
    }
#if TEST_CODE
    else if (params.functionModeChoice == 2) {
        cout << "TEST SOM" << endl;

        GridDensity gd;
        Grid2DPoints gm;

        // Create operator
        SomOperator<MatDensity, Grid2DPoints>* somOp;
        somOp = new SomOperator<MatDensity, Grid2DPoints>();
        somOp->initialize(&gd, &gm, &params, &os);

        // Execution
        somOp->init();
        while (somOp->activate());
        somOp->run();

        // Create output file and save result
        delete somOp;
    }
    else if (params.functionModeChoice == 3) {
        cout << "TEST PROGRAM" << endl;

        // Flux de sortie ouvert pour statistiques
        ofstream os;
        // Create output stream
        os.open(fileStats, ios::app);

        // For Reading
        ifstream fi;

        // Read density map from image (.pgm)
        // need to use Qt as in som_cuda...
        // ...
        // Read density map
        GridDensity gd;
        fi.open(fileData);
        if (!fi) {
            std::cout << "erreur ouverture GridDensity" << endl;
            exit(0);
        }
        else
            fi >> gd;
        fi.close();

        // Read grid of neurons
        Grid2DPoints gm;
        fi.open(fileSolution);
        if (!fi) {
            std::cout << "erreur ouverture Grid2DPoints" << endl;
            exit(0);
        }
        else
            fi >> gm;
        fi.close();

        // Create operator
        SomOperator<MatDensity, Grid2DPoints>* somOp;
        somOp = new SomOperator<MatDensity, Grid2DPoints>();
        somOp->initialize(&gd, &gm, &params, &os);

        // Execution
        somOp->init();
        while (somOp->activate());
        somOp->run();

        // Create output file and save result
        delete somOp;
    }
    else    if (params.functionModeChoice == 4) {
        cout << "INSTANCE GENERATOR" << endl;

        InstanceGenerator* iG = NULL;
        iG = new InstanceGenerator();
        iG->initialize(fileData, fileSolution, fileStats, &params);
        iG->run();

        delete iG;
    }
#endif
    return 0;
}//main

