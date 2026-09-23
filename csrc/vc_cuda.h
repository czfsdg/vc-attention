#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace vc {

struct Shape {
    int heads, nq, nk, d, dv;
    int bq, bk, k_quant;
    int pq, pk, dp, vp;  // Padded sequence lengths and channel dimensions.
    int nq_tiles, nk_tiles, nk_quant_tiles;
    bool smooth, expcast;
    int mean_type;  // 0=FP32, 1=FP16, 2=BF16.
    float scale;
};

struct Buffers {
    uint8_t *q8, *k8, *v8, *p8;
    float *qs, *ks, *vs, *mean;
    float *scores, *pv, *maximum, *denom, *alpha, *mass, *accum;
    float *output;
    void *workspace;
    size_t workspace_bytes;
};

// Inputs/outputs and scratch buffers belong to PyTorch's allocator. All work
// uses the caller's current stream; no cudaDeviceSynchronize or default stream.
void forward(const float* q, const float* k, const float* v,
             const Shape& s, const Buffers& b,
             cublasLtHandle_t handle, cudaStream_t stream);
void expcast(const float* input, uint8_t* output, int64_t size, cudaStream_t stream);

}  // namespace vc
