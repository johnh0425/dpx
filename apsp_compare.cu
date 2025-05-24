// File: apsp_compare.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <climits>

#define INF 1000000000
#define M 300  // Matrix size (300x300)

__device__ __host__ inline int pack_short2(short a, short b) {
    return ((int)b << 16) | (a & 0xFFFF);
}

__device__ __host__ inline short unpack_low(int val) {
    return (short)(val & 0xFFFF);
}

__device__ __host__ inline short unpack_high(int val) {
    return (short)((val >> 16) & 0xFFFF);
}

__global__ void test_kernel() {
    printf("Hello from thread (%d, %d)\\n", threadIdx.x, threadIdx.y);
}

__device__ int dpx_fallback(int a, int b, int c) {
    short a0 = unpack_low(a), a1 = unpack_high(a);
    short b0 = unpack_low(b), b1 = unpack_high(b);
    short c0 = unpack_low(c), c1 = unpack_high(c);
    short r0 = max(0, min(a0, b0 + c0));
    short r1 = max(0, min(a1, b1 + c1));
    return pack_short2(r0, r1);
}

__global__ void floyd_warshall_standard(int *mat) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= M || j >= M) return;

    for (int k = 0; k < M; k++) {
        int ij = i * M + j;
        int ik = i * M + k;
        int kj = k * M + j;
        int a = mat[ik], b = mat[kj];
        if (a < INF && b < INF)
            mat[ij] = min(mat[ij], a + b);
        __syncthreads();
    }

    if (i == 0 && j == 0) {
        printf("D[0][0] = %d\\n", mat[0]);
    }
    
}

__global__ void floyd_warshall_dpx(int *mat_packed) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= M || j >= M / 2) return;

    for (int k = 0; k < M; k++) {
        int ij = i * (M / 2) + j;
        int ik = i * (M / 2) + k / 2;
        int kj = k * (M / 2) + j;
        int dij = mat_packed[ij];
        int dik = mat_packed[ik];
        int dkj = mat_packed[kj];

#if __CUDA_ARCH__ >= 900
        int updated = __viaddmin_s16x2_relu(dij, dik, dkj);
#else
        int updated = dpx_fallback(dij, dik, dkj);
#endif

        mat_packed[ij] = updated;
        __syncthreads();
    }
}

void load_csv_to_matrix(const std::string &filename, int *mat, int N) {
    std::ifstream file(filename);
    std::string line;
    int row = 0;
    std::getline(file, line);  // skip header
    while (std::getline(file, line) && row < N) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');  // skip index label
        for (int col = 0; col < N; col++) {
            std::getline(ss, cell, ',');
            int val = std::stoi(cell);
            mat[row * N + col] = (val >= INF) ? INF : val;
        }
        row++;
    }
}

void run_standard(const int *host_input) {
    int *dev_mat;
    size_t bytes = M * M * sizeof(int);
    cudaMalloc(&dev_mat, bytes);
    cudaMemcpy(dev_mat, host_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    floyd_warshall_standard<<<grid, block>>>(dev_mat);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Standard APSP time: %.3f ms\n", ms);
    cudaFree(dev_mat);
}

void run_dpx(const int *host_input) {
    const int M_packed = M / 2;
    int *packed = new int[M * M_packed];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M_packed; j++) {
            short a = host_input[i * M + (2 * j)];
            short b = host_input[i * M + (2 * j + 1)];
            packed[i * M_packed + j] = pack_short2(a, b);
        }
    }

    int *dev_packed;
    size_t bytes = M * M_packed * sizeof(int);
    cudaMalloc(&dev_packed, bytes);
    cudaMemcpy(dev_packed, packed, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M_packed + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    floyd_warshall_dpx<<<grid, block>>>(dev_packed);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("DPX APSP time: %.3f ms\n", ms);

    cudaFree(dev_packed);
    delete[] packed;
}

int main() {
    int *input = new int[M * M];
    load_csv_to_matrix("LaJolla_distance_matrix.csv", input, M);
    run_standard(input);
    run_dpx(input);
    delete[] input;
    return 0;
}
