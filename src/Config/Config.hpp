//
// Created by peter on 23/06/19.
//

#ifndef HW1_CONFIG_HPP
#define HW1_CONFIG_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>

class Config {
public:
    explicit Config(const char *filename);

    void set(const std::string &key, const std::string &value);

    int get_int(const std::string &key) const;
    std::string get_str(const std::string &key) const;
private:
    std::unordered_map<std::string, std::string> config;

    std::string get(const std::string &key) const;
};


#endif //HW1_CONFIG_HPP
