#include "vc_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vc {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void blas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed, cuBLAS status=" + std::to_string(status));
}

__device__ float reduce_max(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ float reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ uint8_t encode(float x) {
    return __nv_cvt_float_to_fp8(fminf(448.0f, fmaxf(-448.0f, x)), __NV_SATFINITE, __NV_E4M3);
}

__device__ float decode_positive(uint8_t code) {
    const int exponent = code >> 3;
    const int mantissa = code & 7;
    return exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
                         : ldexpf(1.0f + mantissa * 0.125f, exponent - 7);
}

__device__ uint8_t expcast_code(float x) {
    // Match the Python eager implementation: multiply and add round separately.
    const float u = __fadd_rn(__fmul_rn(x, 11.541560173034668f), 119.6500015258789f);
    return static_cast<uint8_t>(__float2int_rn(fminf(120.0f, fmaxf(0.0f, u))));
}

__global__ void expcast_kernel(const float* x, uint8_t* y, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(blockDim.x) * gridDim.x) y[i] = expcast_code(x[i]);
}

// A scale belongs to a sequence block, independently for each batch/head.
// D <= 128, hence one channel group per Q/K block, as in the reference.
__global__ void quantize_qk(const float* input, uint8_t* output, float* scales,
                           int n, int padded_n, int d, int padded_d,
                           int tile_size, int tile_count) {
    __shared__ float scratch[256];
    const int head = blockIdx.x / tile_count;
    const int tile = blockIdx.x % tile_count;
    const int start = tile * tile_size;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < tile_size * d; i += blockDim.x) {
        const int row = start + i / d;
        if (row < n) amax = fmaxf(amax, fabsf(input[(int64_t(head) * n + row) * d + i % d]));
    }
    amax = reduce_max(amax, scratch);
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    if (threadIdx.x == 0) scales[head * tile_count + tile] = scale;
    for (int i = threadIdx.x; i < tile_size * padded_d; i += blockDim.x) {
        const int row = start + i / padded_d;
        const int col = i % padded_d;
        if (row < padded_n) {
            const float value = row < n && col < d
                ? input[(int64_t(head) * n + row) * d + col] : 0.0f;
            output[(int64_t(head) * padded_n + row) * padded_d + col] = encode(value / scale);
        }
    }
}

// Store each V tile column-major: [head, kv_tile, channel, token]. This is
// precisely the non-transposed B operand required by FP8 cuBLASLt TN GEMM.
__global__ void prepare_values(const float* input, uint8_t* output, float* scales,
                               float* means, Shape s) {
    __shared__ float scratch[256];
    const int channel = blockIdx.x % s.vp;
    const int tile_head = blockIdx.x / s.vp;
    const int tile = tile_head % s.nk_tiles;
    const int head = tile_head / s.nk_tiles;
    const int start = tile * s.bk;
    const int count = min(s.bk, s.nk - start);
    float sum = 0.0f;
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        if (channel < s.dv)
            sum += input[(int64_t(head) * s.nk + start + row) * s.dv + channel];
    }
    float mean = s.smooth ? reduce_sum(sum, scratch) / count : 0.0f;
    if (s.mean_type == 1) mean = __half2float(__float2half_rn(mean));
    if (s.mean_type == 2) mean = __bfloat162float(__float2bfloat16_rn(mean));
    float amax = 0.0f;
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        if (channel < s.dv)
            amax = fmaxf(amax, fabsf(input[(int64_t(head) * s.nk + start + row) * s.dv + channel] - mean));
    }
    amax = reduce_max(amax, scratch);
    const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
    const int64_t index = int64_t(tile_head) * s.vp + channel;
    if (threadIdx.x == 0) { means[index] = mean; scales[index] = scale; }
    for (int row = threadIdx.x; row < s.bk; row += blockDim.x) {
        const float value = row < count && channel < s.dv
            ? input[(int64_t(head) * s.nk + start + row) * s.dv + channel] - mean : 0.0f;
        output[index * s.bk + row] = encode(value / scale);
    }
}

__global__ void init_state(Buffers b, Shape s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < s.bq) { b.maximum[i] = -CUDART_INF_F; b.denom[i] = 0.0f; }
    if (i < s.bq * s.vp) b.accum[i] = 0.0f;
}

// QK GEMM produces column-major scores. P is written row-major so it can be
// interpreted as a transposed column-major A operand in the next GEMM.
__global__ void softmax_update(Buffers b, Shape s, int head, int q_tile, int k_tile) {
    __shared__ float scratch[256];
    const int row = blockIdx.x;
    const int j = threadIdx.x;
    const bool valid_row = q_tile * s.bq + row < s.nq;
    const bool valid_key = j < s.bk && k_tile * s.bk + j < s.nk;
    const float qs = b.qs[head * s.nq_tiles + q_tile];
    const float ks = b.ks[head * s.nk_quant_tiles + (k_tile * s.bk) / s.k_quant];
    float score = -CUDART_INF_F;
    if (valid_row && valid_key)
        score = ((b.scores[j * s.bq + row] * qs) * ks) * s.scale;
    const float tile_max = reduce_max(score, scratch);
    const float new_max = valid_row ? fmaxf(tile_max, b.maximum[row]) : 0.0f;
    const float alpha = valid_row ? expf(b.maximum[row] - new_max) : 0.0f;
    float probability = 0.0f;
    uint8_t payload = 0;
    if (valid_row && valid_key) {
        const float shifted = score - new_max;
        if (s.expcast) {
            payload = expcast_code(shifted);
            probability = decode_positive(payload) / 256.0f;
        } else {
            probability = expf(shifted);
            payload = encode(probability * 256.0f);
        }
    }
    if (j < s.bk) b.p8[row * s.bk + j] = payload;
    const float mass = reduce_sum(probability, scratch);
    if (j == 0) {
        b.alpha[row] = alpha;
        b.mass[row] = mass;
        b.maximum[row] = new_max;
        b.denom[row] = alpha * b.denom[row] + mass;
    }
}

__global__ void accumulate(Buffers b, Shape s, int head, int k_tile) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.bq * s.vp) return;
    const int row = i / s.vp;
    const int col = i % s.vp;
    const int64_t meta = (int64_t(head) * s.nk_tiles + k_tile) * s.vp + col;
    const float product = b.pv[col * s.bq + row] * (b.vs[meta] / 256.0f);
    b.accum[i] = (b.alpha[row] * b.accum[i] + product) + b.mass[row] * b.mean[meta];
}

__global__ void write_output(Buffers b, Shape s, int head, int q_tile) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.bq * s.dv) return;
    const int row = i / s.dv;
    const int col = i % s.dv;
    const int q = q_tile * s.bq + row;
    if (q < s.nq)
        b.output[(int64_t(head) * s.nq + q) * s.dv + col] = b.accum[row * s.vp + col] / b.denom[row];
}

// Descriptors/heuristics are built twice per invocation, not per tile. Scratch
// is reused only on the same CUDA stream, after its previous consumer.
class Fp8Gemm {
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, c_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    void release() noexcept {
        if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
        if (c_) cublasLtMatrixLayoutDestroy(c_);
        if (b_) cublasLtMatrixLayoutDestroy(b_);
        if (a_) cublasLtMatrixLayoutDestroy(a_);
        if (operation_) cublasLtMatmulDescDestroy(operation_);
    }
public:
    Fp8Gemm(cublasLtHandle_t handle, int m, int n, int k, size_t workspace_bytes) {
        try {
            blas_check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
            cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
            blas_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "TRANSA");
            blas_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)), "TRANSB");
            int8_t fast_accum = 0;
            blas_check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)), "FAST_ACCUM");
            // Scale pointers are unset: FP8 payload products use scale 1.
            // Q/K block and V channel scales are restored by our CUDA kernels.
            blas_check(cublasLtMatrixLayoutCreate(&a_, CUDA_R_8F_E4M3, k, m, k), "A layout");
            blas_check(cublasLtMatrixLayoutCreate(&b_, CUDA_R_8F_E4M3, k, n, k), "B layout");
            blas_check(cublasLtMatrixLayoutCreate(&c_, CUDA_R_32F, m, n, m), "C/D layout");
            blas_check(cublasLtMatmulPreferenceCreate(&preference_), "PreferenceCreate");
            blas_check(cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "workspace preference");
            cublasLtMatmulHeuristicResult_t candidates[8]{};
            int count = 0;
            blas_check(cublasLtMatmulAlgoGetHeuristic(handle, operation_, a_, b_, c_, c_, preference_, 8, candidates, &count), "FP8 heuristic");
            bool found = false;
            for (int i = 0; i < count; ++i) {
                if (candidates[i].state == CUBLAS_STATUS_SUCCESS && candidates[i].workspaceSize <= workspace_bytes) {
                    algorithm_ = candidates[i].algo;
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("No FP8->FP32 cuBLASLt algorithm for this shape/runtime. No fallback was used.");
        } catch (...) { release(); throw; }
    }
    ~Fp8Gemm() { release(); }
    Fp8Gemm(const Fp8Gemm&) = delete;
    Fp8Gemm& operator=(const Fp8Gemm&) = delete;
    void run(cublasLtHandle_t handle, const uint8_t* a, const uint8_t* b,
             float* out, const Buffers& buffers, cudaStream_t stream) const {
        const float alpha = 1.0f, beta = 0.0f;
        blas_check(cublasLtMatmul(handle, operation_, &alpha, a, a_, b, b_,
                                 &beta, out, c_, out, c_, &algorithm_, buffers.workspace,
                                 buffers.workspace_bytes, stream), "FP8 Tensor Core matmul");
    }
};

}  // namespace

void expcast(const float* input, uint8_t* output, int64_t size, cudaStream_t stream) {
    if (!size) return;
    expcast_kernel<<<static_cast<unsigned>(std::min<int64_t>((size + 255) / 256, 65535)), 256, 0, stream>>>(input, output, size);
    cuda_check(cudaGetLastError(), "ExpCast launch");
}

void forward(const float* q, const float* k, const float* v,
             const Shape& s, const Buffers& b, cublasLtHandle_t handle, cudaStream_t stream) {
    Fp8Gemm qk(handle, s.bq, s.bk, s.dp, b.workspace_bytes);
    Fp8Gemm pv(handle, s.bq, s.vp, s.bk, b.workspace_bytes);
    quantize_qk<<<s.heads * s.nq_tiles, 256, 0, stream>>>(q, b.q8, b.qs, s.nq, s.pq, s.d, s.dp, s.bq, s.nq_tiles);
    cuda_check(cudaGetLastError(), "Q quantization launch");
    quantize_qk<<<s.heads * s.nk_quant_tiles, 256, 0, stream>>>(k, b.k8, b.ks, s.nk, s.pk, s.d, s.dp, s.k_quant, s.nk_quant_tiles);
    cuda_check(cudaGetLastError(), "K quantization launch");
    prepare_values<<<s.heads * s.nk_tiles * s.vp, 256, 0, stream>>>(v, b.v8, b.vs, b.mean, s);
    cuda_check(cudaGetLastError(), "V preprocessing launch");
    for (int head = 0; head < s.heads; ++head) {
        for (int qi = 0; qi < s.nq_tiles; ++qi) {
            init_state<<<(s.bq * s.vp + 255) / 256, 256, 0, stream>>>(b, s);
            for (int ki = 0; ki < s.nk_tiles; ++ki) {
                const uint8_t* qt = b.q8 + (int64_t(head) * s.pq + qi * s.bq) * s.dp;
                const uint8_t* kt = b.k8 + (int64_t(head) * s.pk + ki * s.bk) * s.dp;
                qk.run(handle, qt, kt, b.scores, b, stream);
                softmax_update<<<s.bq, 256, 0, stream>>>(b, s, head, qi, ki);
                cuda_check(cudaGetLastError(), "Softmax/ExpCast launch");
                const uint8_t* vt = b.v8 + (int64_t(head) * s.nk_tiles + ki) * s.vp * s.bk;
                pv.run(handle, b.p8, vt, b.pv, b, stream);
                accumulate<<<(s.bq * s.vp + 255) / 256, 256, 0, stream>>>(b, s, head, ki);
                cuda_check(cudaGetLastError(), "online output update launch");
            }
            write_output<<<(s.bq * s.dv + 255) / 256, 256, 0, stream>>>(b, s, head, qi);
            cuda_check(cudaGetLastError(), "output normalization launch");
        }
    }
}

}  // namespace vc
