/*
 ***************************************************************************
 *
 * Auteur : J.C. Creput
 * Date creation : janvier 2015
 *
 ***************************************************************************
 */
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>

#include "ConfigParams.h"

/*! \brief Foncteur de conversion string vers booleen
 */
struct ToBool
{
    bool operator()(std::string const &str) {
        return str.compare("true") == 0;
    }
} toBool;

ConfigParams::ConfigParams(char* file) : configFile(file){

    if (file == NULL)
        configFile = "config.cfg";

    initDefaultParameters();
}

void ConfigParams::initDefaultParameters() {

    // Parametres generaux
    functionModeChoice = 0;

}//initDefaultParameters

void ConfigParams::readConfigParameters() {

    ConfigFile cf(configFile);

    // Global Parameters
    functionModeChoice = (int) cf.Value("global_param","functionModeChoice", functionModeChoice);

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
