#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Config/Config.hpp"

#include "data.hpp"
#include "utils.hpp"

// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

static void handleCUDAError(
        cudaError_t error,        // error code
        const char *file,         // file within error was generated
        int line)                 // line where error occurs
{
    if (error != cudaSuccess) {    // any error -> display error message and terminate application
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_ERROR(error) ( handleCUDAError( error, __FILE__, __LINE__ ) )

#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

// Filtering

__global__ void cuda_filter(Data *input, int *filter, int elements_per_block, int interval_from, int interval_to) {
    int bid = blockIdx.x;
    int block_offset = bid * elements_per_block;

    int tid = threadIdx.x;

    filter[block_offset + tid] = FILTER(input[block_offset + tid], interval_from, interval_to);
}

void perform_filter(Data *input, int *filter, int threads_per_block, int size, int interval_From, int interval_to) {
    int n_blocks = size / threads_per_block;

    size_t input_size = size * sizeof(Data);
    size_t filter_size = size * sizeof(int);

    Data *input_cuda;
    int *filter_cuda;

    CHECK_ERROR(cudaMalloc((void **) &input_cuda, input_size));
    CHECK_ERROR(cudaMalloc((void **) &filter_cuda, filter_size));

    CHECK_ERROR(cudaMemcpy(input_cuda, input, input_size, cudaMemcpyHostToDevice));

    cuda_filter <<< n_blocks, threads_per_block >>> (input_cuda, filter_cuda, threads_per_block, interval_From, interval_to);

    int remains = size - n_blocks * threads_per_block;

    if (remains > 0) {
        cuda_filter <<< 1, remains >>> (&(input_cuda[size - remains]), &(filter_cuda[size - remains]), remains, interval_From, interval_to);
    }

    CHECK_ERROR(cudaMemcpy(filter, filter_cuda, filter_size, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(input_cuda));
    CHECK_ERROR(cudaFree(filter_cuda));
}

// Adding

__global__ void add(int *output, int length, int *n) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_offset = bid * length;

    output[block_offset + tid] += n[bid];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_offset = bid * length;

    output[block_offset + tid] += n1[bid] + n2[bid];
}

// Scanning

__global__ void cuda_scan_small(int *filter, int *scan, int size, int size_pot) {
    extern __shared__ int sh_data[];

    int thid = threadIdx.x;

    int ai = thid;
    int bi = thid + (size / 2);

    int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
    int bank_offset_b = CONFLICT_FREE_OFFSET(bi);

    if (thid < size) {
        sh_data[ai + bank_offset_a] = filter[ai];
        sh_data[bi + bank_offset_b] = filter[bi];
    } else {
        sh_data[ai + bank_offset_a] = 0;
        sh_data[bi + bank_offset_b] = 0;
    }

    int offset = 1;

    for (int d = size_pot >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            sh_data[bi] += sh_data[ai];
        }

        offset *= 2;
    }

    if (thid == 0) {
        sh_data[size_pot - 1 + CONFLICT_FREE_OFFSET(size_pot - 1)] = 0;
    }

    for (int d = 1; d < size_pot; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = sh_data[ai];
            sh_data[ai] = sh_data[bi];
            sh_data[bi] += t;
        }
    }

    __syncthreads();

    if (thid < size) {
        scan[ai] = sh_data[ai + bank_offset_a];
        scan[bi] = sh_data[bi + bank_offset_b];
    }
}

__global__ void cuda_scan_large(int *filter, int *scan, int size, int *sums) {
    extern __shared__ int sh_data[];

    int bid = blockIdx.x;
    int thid = threadIdx.x;
    int block_offset = bid * size;

    int ai = thid;
    int bi = thid + (size / 2);

    int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
    int bank_offset_b = CONFLICT_FREE_OFFSET(bi);

    sh_data[ai + bank_offset_a] = filter[block_offset + ai];
    sh_data[bi + bank_offset_b] = filter[block_offset + bi];

    int offset = 1;

    for (int d = size >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            sh_data[bi] += sh_data[ai];
        }

        offset *= 2;
    }

    __syncthreads();

    if (thid == 0) {
        sums[bid] = sh_data[size - 1 + CONFLICT_FREE_OFFSET(size - 1)];
        sh_data[size - 1 + CONFLICT_FREE_OFFSET(size - 1)] = 0;
    }

    for (int d = 1; d < size; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = sh_data[ai];
            sh_data[ai] = sh_data[bi];
            sh_data[bi] += t;
        }
    }

    __syncthreads();

    scan[block_offset + ai] = sh_data[ai + bank_offset_a];
    scan[block_offset + bi] = sh_data[bi + bank_offset_b];
}

// Only declaration, because of scan_large <-> scan_large_even
void scan_large(int *filter_cuda, int *scan_cuda, int size, int elements_per_block, int threads_per_block);

void scan_small(int *filter_cuda, int *scan_cuda, int size) {
    int pot = next_power_of_two(size);

    cuda_scan_small <<< 1, (size + 1) / 2, 2 * pot * sizeof(int) >>> (filter_cuda, scan_cuda, size, pot);
}

void scan_large_even(int *filter_cuda, int *scan_cuda, int size, int elements_per_block, int threads_per_block) {
    int n_blocks = size / elements_per_block;

    size_t int_size = elements_per_block * sizeof(int);

    int *sums;
    int *incr;

    CHECK_ERROR(cudaMalloc((void **) &sums, n_blocks * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **) &incr, n_blocks * sizeof(int)));

    cuda_scan_large <<< n_blocks, threads_per_block, 2 * int_size >>> (filter_cuda, scan_cuda, elements_per_block, sums);

    int sums_threads = (n_blocks + 1) / 2;

    if (sums_threads > threads_per_block) {
        scan_large(sums, incr, n_blocks, elements_per_block, threads_per_block);
    } else {
        scan_small(sums, incr, n_blocks);
    }

    add <<< n_blocks, elements_per_block >>> (scan_cuda, elements_per_block, incr);

    CHECK_ERROR( cudaFree(incr) );
    CHECK_ERROR( cudaFree(sums) );
}

void scan_large(int *filter_cuda, int *scan_cuda, int size, int elements_per_block, int threads_per_block) {
    int remainder = size % elements_per_block;

    if (remainder == 0) {
        scan_large_even(filter_cuda, scan_cuda, size, elements_per_block, threads_per_block);
    }

    else {
        int size_multiple = size - remainder;

        scan_large_even(filter_cuda, scan_cuda, size_multiple, elements_per_block, threads_per_block);

        int *start_of_output = &(scan_cuda[size_multiple]);

        scan_small(&(filter_cuda[size_multiple]), start_of_output, remainder);

        add <<< 1, remainder >>> (start_of_output, remainder, &(filter_cuda[size_multiple - 1]), &(scan_cuda[size_multiple - 1]));
    }
}

void perform_scan(int *filter, int *scan, int threads_per_block, int size) {
    int elements_per_block = 2 * threads_per_block;

    int *filter_cuda;
    int *scan_cuda;

    size_t int_size = size * sizeof(int);

    CHECK_ERROR(cudaMalloc((void **) &filter_cuda, int_size));
    CHECK_ERROR(cudaMalloc((void **) &scan_cuda, int_size));

    CHECK_ERROR(cudaMemcpy(filter_cuda, filter, int_size, cudaMemcpyHostToDevice));

    if (size > elements_per_block) {
        scan_large(filter_cuda, scan_cuda, size, elements_per_block, threads_per_block);
    }

    else {
        scan_small(filter_cuda, scan_cuda, size);
    }

    CHECK_ERROR(cudaMemcpy(scan, scan_cuda, int_size, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(filter_cuda));
    CHECK_ERROR(cudaFree(scan_cuda));
}

// Truncating

__global__ void cuda_truncate(Data *input, int *filter, int *scan, Data *filtered, int elements_per_block) {
    int bid = blockIdx.x;
    int block_offset = bid * elements_per_block;

    int tid = threadIdx.x;

    if (filter[block_offset + tid] == 1) {
        filtered[scan[block_offset + tid]] = input[block_offset + tid];
    }
}

void perform_truncate(Data *input, int *filter, int *scan, Data *filtered, int threads_per_block, int size, int new_size) {
    int n_blocks = size / threads_per_block;

    Data *input_cuda;
    int *filter_cuda;
    int *scan_cuda;
    Data *filtered_cuda;

    size_t data_size = size * sizeof(Data);
    size_t int_size = size * sizeof(int);
    size_t filtered_size = new_size * sizeof(Data);

    CHECK_ERROR(cudaMalloc((void **) &input_cuda, data_size));
    CHECK_ERROR(cudaMalloc((void **) &filter_cuda, int_size));
    CHECK_ERROR(cudaMalloc((void **) &scan_cuda, int_size));
    CHECK_ERROR(cudaMalloc((void **) &filtered_cuda, filtered_size));

    CHECK_ERROR(cudaMemcpy(input_cuda, input, data_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(filter_cuda, filter, int_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(scan_cuda, scan, int_size, cudaMemcpyHostToDevice));

    cuda_truncate <<< n_blocks, threads_per_block >>> (input_cuda, filter_cuda, scan_cuda, filtered_cuda, threads_per_block);

    int remains = size - n_blocks * threads_per_block;

    if (remains > 0) {
        cuda_truncate <<< 1, remains >>> (&(input_cuda[size - remains]), &(filter_cuda[size - remains]), &(scan_cuda[size - remains]), filtered_cuda, remains);
    }

    CHECK_ERROR(cudaMemcpy(filtered, filtered_cuda, filtered_size, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(input_cuda));
    CHECK_ERROR(cudaFree(filter_cuda));
    CHECK_ERROR(cudaFree(scan_cuda));
    CHECK_ERROR(cudaFree(filtered_cuda));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        EXIT("Usage: ./CSC path_to_config_file")
    }

    // Constants from config

    const auto config = Config(argv[1]);

    const auto SIZE = config.get_int("N");
    const auto THREADS_PER_BLOCK = config.get_int("THREADS_PER_BLOCK");

    if (THREADS_PER_BLOCK < 1) {
        EXIT("At least one THREADS_PER_BLOCK required")
    }

    const auto KEY_FROM = config.get_int("KEY_FROM");
    const auto KEY_TO = config.get_int("KEY_TO");

    const auto INTERVAL_FROM = config.get_int("INTERVAL_FROM");
    const auto INTERVAL_TO = config.get_int("INTERVAL_TO");

    // Generate input data

    Data *input;

    input = (Data *) malloc(SIZE * sizeof(Data));

    generate(input, SIZE, KEY_FROM, KEY_TO);

    // Perform filter

    int *filter;

    filter = (int *) malloc(SIZE * sizeof(int));

    perform_filter(input, filter, THREADS_PER_BLOCK, SIZE, INTERVAL_FROM, INTERVAL_TO);

    // Perform scan

    int *scan;

    scan = (int *) malloc(SIZE * sizeof(int));

    perform_scan(filter, scan, THREADS_PER_BLOCK, SIZE);

    // Get new size

    int filtered_size;

    if (scan[SIZE - 1] == 0 && !FILTER(input[SIZE - 1], INTERVAL_FROM, INTERVAL_TO)) {
        filtered_size = 0;
    }

    else if (scan[SIZE - 1] > 0 && FILTER(input[SIZE - 1], INTERVAL_FROM, INTERVAL_TO)) {
        filtered_size = scan[SIZE - 1] + 1;
    }

    else {
        filtered_size = scan[SIZE - 1];
    }

    // Perform truncation

    Data *filtered;

    filtered = (Data *) malloc(filtered_size * sizeof(Data));

    perform_truncate(input, filter, scan, filtered, THREADS_PER_BLOCK, SIZE, filtered_size);

    // Just print

    print("Input", input, SIZE);
    print("Filter", filter, SIZE);
    print("Scan", scan, SIZE);
    print("Final", filtered, filtered_size);

    std::cout << std::endl << "Filtered out " << SIZE - filtered_size << " items from " << SIZE << std::endl;

    // Free host memory

    free(input);
    free(filtered);
    free(filter);
    free(scan);
}