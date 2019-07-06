//
// Created by peter on 23/06/19.
//

#include "Config.hpp"

Config::Config(const char *filename) {
    std::ifstream is_file(filename);

    std::string key, line, value;
    while (std::getline(is_file, line)) {
        std::istringstream is_line(line);

        if (std::getline(is_line, key, '=')) {
            if (std::getline(is_line, value)) {
                this->set(key, value);
            }
        }
    }
}

void Config::set(const std::string &key, const std::string &value) {
    this->config[key] = value;
}

std::string Config::get(const std::string &key) const {
    return this->config.at(key);
}

int Config::get_int(const std::string &key) const {
    return std::stoi(this->get(key));
}

std::string Config::get_str(const std::string &key) const {
    return this->get(key);
}