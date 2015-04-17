/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : janvier 2015
 *
 ***************************************************************************
 */
#include "InstanceGenerator.h"

#define TEST_CODE   0 // code pour test temporaire (=0 => code supprime)

#define ACCES_APPLICATION "..\\..\\bin\\application.exe "
#define SUFFIXE_DATA ".data"

extern void aleat_initialize(void);
//extern int aleat_int(int min, int max);
extern double aleat_double(double a, double b);

template <typename T> std::string tostr(const T& t) { std::ostringstream os; os<<t; return os.str(); }

struct ToDouble
{
    double operator()(std::string const &str) { return boost::lexical_cast<double>(str.c_str()); }
};

struct ToInt
{
    int operator()(std::string const &str) { return boost::lexical_cast<int>(str.c_str()); }
};

void InstanceGenerator::initialize(char* data, char* sol, char* stats, ConfigParams* params) {
    this->params = params;
    initialize(data, sol, stats);
}

void InstanceGenerator::initialize(char* data, char* sol, char* stats) {
    fileData = data;
    fileSolution = sol;
    fileStats = stats;

    initialize();
}

void InstanceGenerator::initialize() {

    initStatistics();
    cout << "GENERATION D'INSTANCES" << endl;

    readDocXmlFromFile(fileData);
    docXmlOutput->setContent(docXml->toByteArray());
    parseDomXml(docXml, inputModeles);
}

void InstanceGenerator::init() {

    saturation = 100;//bidon
    outputVector.clear();
}

void InstanceGenerator::generate(int no_test) {

    // DO IT
    // ...

    // Statistiques
    writeStatistics(no_test);
    cout << "Taux de saturation (%) : "
         << saturation
         << endl;
}//generate

void InstanceGenerator::run() {

    char buffer[256];
    char file_test[256];

    // Generation fichier script de test .bat
    ofstream fo;
    fo.open("test_g.bat");
    ofstream fo2;
    fo2.open("evaluate_g.bat");

    // Generation des instances
    int no_test = 0;
    for (no_test = 0; no_test < params->nInstances; ++no_test)
    {
        // init buffer de sortie
        init();

        cout << "Instance " << no_test << " " << endl;

        // Generation de l'instance
        generate(no_test);

        // Mise à jour output XML
        updateDomXml(docXmlOutput, outputVector);

        // Construction du nom de fichier de sortie
        char* pos = file_test;
        strcpy(file_test, fileSolution);
        if ((pos = strstr(file_test, SUFFIXE_DATA)) != NULL)
            *pos = '\0';
        strcat(file_test, "_");
        strcpy(buffer, tostr(no_test).c_str());
        //itoa(no_test, buffer, 10);
        strcat(file_test, buffer);
        strcat(file_test, SUFFIXE_DATA);

        // Sauvegarde de la sortie (test genere)
        writeDocXmlToFile(file_test, docXmlOutput);
        // Generation fichier script de test .bat
        fo << ACCES_APPLICATION << file_test << " g_output_" << file_test << " g_output.stats config.cfg" << endl;
        fo2 << ACCES_APPLICATION << "g_output_" << file_test << " g_output_" << file_test << " g_output.stats config_evaluate.cfg" << endl;
    }

    cout << "Duree(ms) : " << (clock() - x0)/CLOCKS_PER_SEC << endl;

    closeStatistics();
    // Fichier Script
    fo.close();
    fo2.close();
}//run

void InstanceGenerator::readDocXmlFromFile(char* fileName) {

    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        cout << "INPUT ABSENT : FATAL ERROR !! " << fileName << endl;
        exit(-1);
    }

    docXml->setContent(&file, false);

    file.close();
}//readDocXmlFromFile

void InstanceGenerator::writeDocXmlToFile(char* fileName, QDomDocument* docXml) {

    QFile file(fileName);
    file.open(QIODevice::WriteOnly);

    QTextStream ts(&file);
    ts << *docXml;

    file.close();
}//writeDocXmlToFile

void InstanceGenerator::parseDomXml(QDomDocument* docXml, vector<int>& inputModeles) {
    QDomElement racine = docXml->documentElement();
    QDomElement child = racine.firstChildElement();
    QDomElement next_child;

    /*
     * Premiere phase
     */
    //Boucle permettant la navigation dans le fichier XML
    while(!child.isNull()) {
        next_child = child.nextSiblingElement();
        if (child.tagName() == "symbol") {
            // ...
        }//if
        child = next_child;
    }//while

}//parseDomXml

void InstanceGenerator::updateDomXml(QDomDocument* docXml, vector<int>& outputVector) {
    QDomElement racine = docXml->documentElement();

}

void InstanceGenerator::extractContour(std::string& std_str, Polygon_2& p) {
            // Parser dans un vecteur
            std::vector<double> v;
            boost::char_separator<char> sep(" ,");
            boost::tokenizer<boost::char_separator<char> > tok(std_str, sep);
            std::transform(tok.begin(), tok.end(), std::back_inserter(v), ToDouble());

            // Construire vecteur de Point_2
            std::vector<Point_2> v2;
            for (std::vector<double>::iterator it = v.begin() ; it != v.end(); it+=2) {
                v2.push_back(Point_2(*it, *(it+1)));
            }

            // Construire Polygon_2
            p = Polygon_2(v2.begin(), v2.end());
}//extractContour

void InstanceGenerator::initStatistics() {
    // Ouverture du fichier traitement en mode append
    OutputStream = new ofstream;
    OutputStream->open(fileStats, ios::app);
    if (!OutputStream->rdbuf()->is_open())
    {
        cerr << "Unable to open file " << fileStats << "CRITICAL ERROR" << endl;
        exit(-1);
    }

    initHeaderStatistics(*OutputStream);

    time(&t0);
    x0 = clock();
}

void InstanceGenerator::initHeaderStatistics(std::ostream& o) {
    o  	<< "iteration" << "\t"
        << "taux_de_saturation(%)" << "\t"
        << "duree(ms)" << endl;
}

void InstanceGenerator::writeStatistics(int iteration) {

    writeStatistics(iteration, *OutputStream);

}

void InstanceGenerator::writeStatistics(int iteration, std::ostream& o) {

    o  	<< iteration << "\t"
        << saturation << "\t"
        << (clock() - x0)/CLOCKS_PER_SEC << endl;
}

void InstanceGenerator::closeStatistics() {
    OutputStream->close();
}


