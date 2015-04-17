#ifndef CONFIG_PARAMS_H
#define CONFIG_PARAMS_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput, A. Mansouri, H. Wang
 * Date creation : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <map>
#include <sstream>
#include <cstdlib>

#include "lib_global.h"
#include "random_generator.h"

using namespace std;

//! Version courante
#define VERSION_APPLICATION "version 1.1"

/*! \brief Foncteur de conversion string vers booleen
 */
// struct ToBool
//{
//     bool operator()(std::string const &str) {
//     return str.compare("true") == 0;
//    }
//}toBool;


inline std::string trim(std::string const& source, char const* delims = " \t\r\n") {
    std::string result(source);
    std::string::size_type index = result.find_last_not_of(delims);
    if(index != std::string::npos)
        result.erase(++index);

    index = result.find_first_not_of(delims);
    if(index != std::string::npos)
        result.erase(0, index);
    else
        result.erase();
    return result;
}

class Chameleon {

private:
    std::string value_;

public:

    Chameleon() {}

    explicit Chameleon(std::string const& value) {
        value_=value;
    }
    explicit Chameleon(double d) {
        std::stringstream s;
        s<<d;
        value_=s.str();
    }
    explicit Chameleon(int i) {
        std::stringstream s;
        s<<i;
        value_=s.str();
    }
    explicit Chameleon(bool b) {
        std::stringstream s;
        s<< b ? "true" : "false";
        value_=s.str();
    }

    Chameleon(Chameleon const& other) {
        value_=other.value_;
    }
    Chameleon& operator=(Chameleon const& other) {
        value_=other.value_;
        return *this;
    }
    Chameleon& operator=(std::string const& s) {
        value_=s;
        return *this;
    }
    Chameleon& operator=(double i) {
        std::stringstream s;
        s << i;
        value_ = s.str();
        return *this;
    }

public:

    operator std::string() const {
        return value_;
    }
    operator double() const {
        return atof(value_.c_str());
    }
    operator int() const {
        return atoi(value_.c_str());
    }

    //  operator bool() const {
    //    return ToBool()(value_.c_str());
    //  }

};

class ConfigFile {
    std::map<std::string,Chameleon> content_;

public:
    ConfigFile() {}

    void initialize_configFile(std::string const& configFile)
    {
        std::ifstream file(configFile.c_str());

        std::string line;
        std::string name;
        std::string value;
        std::string inSection;
        int posEqual;
        while (std::getline(file,line)) {

            if (!line.length()) continue;

            if (line[0] == '#') continue;
            if (line[0] == ';') continue;

            if (line[0] == '[') {
                inSection=trim(line.substr(1,line.find(']')-1));
                continue;
            }

            posEqual=line.find('=');
            name  = trim(line.substr(0,posEqual));
            value = trim(line.substr(posEqual+1));

            content_[inSection+'/'+name]=Chameleon(value);
        }
    }
    Chameleon const& Value(std::string const& section, std::string const& entry) const {

        std::map<std::string,Chameleon>::const_iterator ci = content_.find(section + '/' + entry);

        if (ci == content_.end()) {

            throw "does not exist";
        }
        return ci->second;
    }
    Chameleon const& Value(std::string const& section, std::string const& entry, double value) {
        try {
            return Value(section, entry);
        } catch(const char *s) {
            cout << s << " " << entry << endl;
            return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
        }
    }
    Chameleon const& Value(std::string const& section, std::string const& entry, int value) {
        try {
            return Value(section, entry);
        } catch(const char *s) {
            cout << s << " " << entry << endl;
            return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
        }
    }
    Chameleon const& Value(std::string const& section, std::string const& entry, bool value) {
        try {
            return Value(section, entry);
        } catch(const char *s) {
            cout << s << " " << entry << endl;
            return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
        }
    }
    Chameleon const& Value(std::string const& section, std::string const& entry, std::string const& value) {
        try {
            return Value(section, entry);
        } catch(const char *s) {
            cout << s << " " << entry << endl;
            return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
        }
    }
};

class ParamException: public exception
{
    virtual const char* what() const throw()
    {
        return "does not exist";
    }
};

//class LIBSHARED_EXPORT ConfigParams {
class ConfigParams {
    //! Nom du fichier de configuration

    string configFile;
    ConfigFile cf;

public:

    ConfigParams() {}

    ConfigParams(char* file) : configFile(file){

        if (file == NULL)
            configFile = "config.cfg";

        cf.initialize_configFile(configFile);
        initDefaultParameters();
    }

    ~ConfigParams() {}

    int functionModeChoice;
    int levelRadius;

    //! Valeurs des paramètres par défaut

    void initDefaultParameters() {

        // Parametres generaux
        functionModeChoice = 0;
        levelRadius = 25;

    }//initDefaultParameters    //! Fonction de lecture des paramètres
    void readConfigParameters() {
        // Global Parameters
        functionModeChoice = (int) cf.Value("global_param","functionModeChoice", functionModeChoice);
        levelRadius = (int) cf.Value("global_param","levelRadius", levelRadius);

    }//readConfigParameters

    //! Read individual parameter
    void readConfigParameter(std::string const& section, std::string const& entry, int& value) {
        value = (int) cf.Value(section, entry, value);
    }
//    void readConfigParameter(std::string const& section, std::string const& entry, bool& value) {
//        value = (bool) cf.Value(section, entry, value);
//    }
    void readConfigParameter(std::string const& section, std::string const& entry, double& value) {
        value = (double) cf.Value(section, entry, value);
    }
    void readConfigParameter(std::string const& section, std::string const& entry, string& value) {
        value = (string) cf.Value(section, entry, value);
    }

    // Global Parameters

    //! \brief Choix du mode de fonctionnement 0:evaluation, 1:local search,
    //! 2:genetic algorithm, 3:construction initiale seule,
    //! 4:generation automatique d'instances

};

#endif // CONFIG_PARAMS_H
