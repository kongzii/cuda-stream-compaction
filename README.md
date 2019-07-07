# Stream compaction using CUDA

## Task

Given randomly generated input array and filter function: element -> 0 / 1, create compacted array containing only elements from input that passed the function.

Elements are of type `Data`:

```c++
struct Data {
    int key;
    float data;
};
```

`key` is generated randomly in interval specified in `config.cfg`, `FILTER` function returns 0 or 1. 

1 if key is in range `[INTERVAL_FROM, INTERVAL_TO]`, 0 otherwise.

## Algorithm

Actual algorithm performs several steps:

    1. Create filter array -> create new array of 0 and 1, each corresponding to the filter function output for every element
    2. Create scan array -> scan filter array
    3. Based on filter and scan, allocate new truncated array and assign elements that passed filter function from original one

## Requirements

```
cmake
c++
gcc-6
latest nvidia drivers
nvidia-cuda-toolkit
```

## Thanks to

https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

https://github.com/mattdean1/cuda

