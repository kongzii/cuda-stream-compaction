#ifndef HW2_UTILS_HPP
#define HW2_UTILS_HPP

#include "data.hpp"
#include <iostream>

#define EXIT(text) std::cout << text << std::endl; exit(-1);

int next_power_of_two(int x) {
    // https://stackoverflow.com/a/12506181

    int power = 1;

    while (power < x) {
        power *= 2;
    }

    return power;
}

void print(const std::string &name, const int *data, const int size) {
    std::cout << name << ":\t";

    for (int i = 0; i < size; ++i) {
        if (data[i] < 10) {
            std::cout << " ";
        }

        if (data[i] < 100) {
            std::cout << " ";
        }

        std::cout << data[i] << " ";
    }

    std::cout << std::endl;
}

void print(const std::string &name, const Data *data, const int size) {
    std::cout << name << ":\t";

    for (int i = 0; i < size; ++i) {
        if (data[i].key < 10) {
            std::cout << " ";
        }

        if (data[i].key < 100) {
            std::cout << " ";
        }

        std::cout << data[i].key << " ";
    }

    std::cout << std::endl;
}

#endif
