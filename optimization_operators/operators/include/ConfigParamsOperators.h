#ifndef CONFIG_PARAMS_OPERATORS_H
#define CONFIG_PARAMS_OPERATORS_H
/*
 ***************************************************************************
 *
 * Author : J.C. Creput
 * Creation date : Jan. 2015
 *
 ***************************************************************************
 */
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <map>

#include "lib_global.h"
#include "random_generator.h"

using namespace std;

//! Version courante
#define VERSION_APPLICATION "version 1.1"

namespace operators
{

class Chameleon {
public:
  Chameleon() {}
  explicit Chameleon(const std::string&);
  explicit Chameleon(double);
  explicit Chameleon(int);
  explicit Chameleon(bool);

  Chameleon(const Chameleon&);
  Chameleon& operator=(Chameleon const&);

  Chameleon& operator=(std::string const&);
  Chameleon& operator=(double);

public:
  operator std::string() const;
  operator double     () const;
  operator int        () const;
  operator bool       () const;
private:
  std::string value_;
};

class ConfigFile {
  std::map<std::string,Chameleon> content_;

public:
  ConfigFile(std::string const& configFile);

  Chameleon const& Value(std::string const& section, std::string const& entry) const;
  Chameleon const& Value(std::string const& section, std::string const& entry, double value);
  Chameleon const& Value(std::string const& section, std::string const& entry, int value);
  Chameleon const& Value(std::string const& section, std::string const& entry, bool value);
  Chameleon const& Value(std::string const& section, std::string const& entry, std::string const& value);
};

class ConfigParams {
    //! Nom du fichier de configuration
    string configFile;
    //! Configuration content
    ConfigFile cf;

public:
    ConfigParams();
    ConfigParams(char* file);

    //! Valeurs des paramètres par défaut
    void initDefaultParameters();
    //! Fonction de lecture des paramètres
    void readConfigParameters();

    //! Read individual parameter
    void readConfigParameter(std::string const& section, std::string const& entry, int& value) {
        value = (int) cf.Value(section, entry, value);
    }
    void readConfigParameter(std::string const& section, std::string const& entry, bool& value) {
        value = (bool) cf.Value(section, entry, value);
    }
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
    int functionModeChoice;

    //! Affichage trace d'exection statistique
    bool traceActive;
    //! Trace d'exection statistique avec la meilleure solution rencontree, ou la solution courante
    bool traceReportBest;
    //! Trace d'exection avec sauvegarde de la solution courante
    bool traceSaveSolutionFile;

    // Generateur automatique d'instances

    //! Nombre d'instances generees
    int nInstances;
    int saturation;
};

/*! \brief Foncteur de conversion string vers booleen
 */
struct ToBool
{
    bool operator()(std::string const &str) {
        return str.compare("true") == 0;
    }
} toBool;

ConfigParams::ConfigParams(char* file) : configFile(file), cf(file){

    if (file == NULL)
        configFile = "config.cfg";

    initDefaultParameters();
}

void ConfigParams::initDefaultParameters() {

    // Parametres generaux

    functionModeChoice = 0;

    traceActive = false;
    traceReportBest = false;
    traceSaveSolutionFile = false;

    // Generateur automatique d'instances

    nInstances = 100;
    saturation = 10;
}//initDefaultParameters

void ConfigParams::readConfigParameters() {

    // Global Parameters
    functionModeChoice = (int) cf.Value("global_param","functionModeChoice", functionModeChoice);

    traceActive = (bool) cf.Value("global_param","traceActive", traceActive);
    traceReportBest = (bool) cf.Value("global_param","traceReportBest", traceReportBest);
    traceSaveSolutionFile = (bool) cf.Value("global_param","traceSaveSolutionFile", traceSaveSolutionFile);

    // Generateur d'instances
    nInstances = (int) cf.Value("instance_generator","nInstances", nInstances);
    saturation = (int) cf.Value("instance_generator","saturation", saturation);

}//readConfigParameters

Chameleon::Chameleon(std::string const& value) {
  value_=value;
}

Chameleon::Chameleon(double d) {
  std::stringstream s;
  s<<d;
  value_=s.str();
}

Chameleon::Chameleon(int i) {
  std::stringstream s;
  s<<i;
  value_=s.str();
}

Chameleon::Chameleon(bool b) {
  std::stringstream s;
  s<< b ? "true" : "false";
  value_=s.str();
}

Chameleon::Chameleon(Chameleon const& other) {
  value_=other.value_;
}

Chameleon& Chameleon::operator=(Chameleon const& other) {
  value_=other.value_;
  return *this;
}

Chameleon& Chameleon::operator=(std::string const& s) {
  value_=s;
  return *this;
}

Chameleon& Chameleon::operator=(double i) {
  std::stringstream s;
  s << i;
  value_ = s.str();
  return *this;
}

Chameleon::operator std::string() const {
  return value_;
}

Chameleon::operator double() const {
  return atof(value_.c_str());
}

Chameleon::operator int() const {
  return atoi(value_.c_str());
}

Chameleon::operator bool() const {
  return ToBool()(value_.c_str());
}

std::string trim(std::string const& source, char const* delims = " \t\r\n") {
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

ConfigFile::ConfigFile(std::string const& configFile) {
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

class ParamException: public exception
{
    virtual const char* what() const throw()
    {
        return "does not exist";
    }
};

Chameleon const& ConfigFile::Value(std::string const& section, std::string const& entry) const {

  std::map<std::string,Chameleon>::const_iterator ci = content_.find(section + '/' + entry);

  if (ci == content_.end()) {

      throw "does not exist";
  }
  return ci->second;
}

Chameleon const& ConfigFile::Value(std::string const& section, std::string const& entry, double value) {
    try {
        return Value(section, entry);
    } catch(const char *s) {
        cout << s << " " << entry << endl;
        return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
    }
}

Chameleon const& ConfigFile::Value(std::string const& section, std::string const& entry, int value) {
    try {
        return Value(section, entry);
    } catch(const char *s) {
        cout << s << " " << entry << endl;
        return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
    }
}

Chameleon const& ConfigFile::Value(std::string const& section, std::string const& entry, bool value) {
    try {
        return Value(section, entry);
    } catch(const char *s) {
        cout << s << " " << entry << endl;
        return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
    }
}

Chameleon const& ConfigFile::Value(std::string const& section, std::string const& entry, std::string const& value) {
    try {
        return Value(section, entry);
    } catch(const char *s) {
        cout << s << " " << entry << endl;
        return content_.insert(std::make_pair(section+'/'+entry, Chameleon(value))).first->second;
    }
}

}//namespace operators

#endif // CONFIG_PARAMS_OPERATORS_H
