//
// Created by peter on 06/07/19.
//

#ifndef HW2_DATA_HPP
#define HW2_DATA_HPP

#include <random>

struct Data {
    int key;
    float data;
};

#define FILTER(data, from, to) ((data.key >= from && data.key <= to) ? 1 : 0)

void generate(Data *data, const int size, const int key_from, const int key_to) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(key_from, key_to);

    for (int i = 0; i < size; ++i) {
        data[i] = {uni(rng), (float) uni(rng) };
    }
}

#endif //HW2_DATA_HPP
