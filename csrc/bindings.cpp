#include <torch/extension.h>
#include <ATen/ops/argsort.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "vc_cuda.h"

namespace {
using at::Tensor;

// Own only the cuBLASLt resource we use. ATen's CUDAContext.h also includes
// cuSPARSE/cuSOLVER development headers, which this extension does not need.
struct LocalLtHandle {
    cublasLtHandle_t value = nullptr;
    int device;
    explicit LocalLtHandle(int device_index) : device(device_index) {
        TORCH_CHECK(cublasLtCreate(&value) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed");
    }
    LocalLtHandle(const LocalLtHandle&) = delete;
    LocalLtHandle& operator=(const LocalLtHandle&) = delete;
    ~LocalLtHandle() {
        // Thread teardown may occur after CUDA has already shut down. Do not
        // throw from a destructor, and release on the handle's original device.
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        if (value) cublasLtDestroy(value);
        if (previous != device) cudaSetDevice(previous);
    }
};

cublasLtHandle_t blaslt_handle(int device) {
    // Destruction synchronizes the device, so keep handles between forward
    // calls. Each host thread/device gets its own handle; stream is passed to
    // every matmul explicitly. The caller has already set its CUDAGuard.
    thread_local std::unordered_map<int, std::unique_ptr<LocalLtHandle>> handles;
    auto& handle = handles[device];
    if (!handle) handle = std::make_unique<LocalLtHandle>(device);
    return handle->value;
}

int small_int(int64_t value, const char* name) {
    TORCH_CHECK(value > 0 && value <= INT_MAX, name, " must fit a positive int32");
    return static_cast<int>(value);
}

void require_hopper(const Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "cuda_fp8 requires CUDA tensors; there is no CPU fallback");
    int major = 0, minor = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, x.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, x.get_device()));
    TORCH_CHECK(major == 9, "cuda_fp8 targets Hopper (H800/H100/H200, compute capability 9.x); got ",
                major, ".", minor);
}

void block_size(int64_t value, const char* name) {
    TORCH_CHECK(value >= 32 && value <= 256 && value % 32 == 0,
                name, " must be a multiple of 32 in [32, 256]");
}

std::vector<Tensor> preprocess_qk(Tensor q, Tensor k, bool k_smooth, bool hadamard) {
    TORCH_CHECK(q.is_cuda() && k.device() == q.device(), "Q/K preprocessing requires one CUDA device");
    const c10::cuda::CUDAGuard guard(q.device());
    const at::NoGradGuard no_grad;
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.numel() && k.numel(), "Expected nonempty [B,H,N,D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3), "Q/K shapes must match");
    const int width = small_int(q.size(3), "head dimension");
    TORCH_CHECK(width <= 128, "Q/K preprocessing supports head dimensions <= 128");
    q = q.to(at::kFloat).contiguous();
    k = k.to(at::kFloat).contiguous();
    if (!k_smooth && !hadamard) return {q, k};
    int out_width = width;
    if (hadamard) {
        out_width = 1;
        while (out_width < width) out_width *= 2;
    }
    const int heads = small_int(q.size(0) * q.size(1), "B*H");
    const int nq = small_int(q.size(2), "Q sequence length");
    const int nk = small_int(k.size(2), "K sequence length");
    small_int(int64_t(heads) * nq, "Q transform grid");
    small_int(int64_t(heads) * nk, "K transform grid");
    auto mean = k_smooth ? k.mean(2, true) : Tensor();
    auto qr = hadamard ? at::empty({q.size(0), q.size(1), nq, out_width}, q.options()) : q;
    auto kr = at::empty({k.size(0), k.size(1), nk, out_width}, k.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (hadamard)
        vc::transform_qk(q.data_ptr<float>(), nullptr, qr.data_ptr<float>(), heads, nq, width, out_width, true, stream);
    vc::transform_qk(k.data_ptr<float>(), k_smooth ? mean.data_ptr<float>() : nullptr,
                     kr.data_ptr<float>(), heads, nk, width, out_width, hadamard, stream);
    return {qr, kr};
}

Tensor forward(Tensor q, Tensor k, Tensor v, int64_t bq, int64_t k_quant,
               int64_t bk, bool smooth, bool expcast, int64_t mean_type, double scale,
               bool k_smooth, bool hadamard) {
    require_hopper(q);
    const c10::cuda::CUDAGuard guard(q.device());
    const at::NoGradGuard no_grad;
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Expected [B,H,N,D] tensors");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "Q/K/V devices must match");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "Q/K/V dtypes must match");
    TORCH_CHECK(q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "Input dtype must be FP32, FP16, or BF16");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
                q.size(1) == k.size(1) && q.size(1) == v.size(1), "MHA batch/head counts must match; GQA is unsupported");
    TORCH_CHECK(q.size(3) == k.size(3) && k.size(2) == v.size(2), "Incompatible Q/K/V shapes");
    TORCH_CHECK(q.size(3) <= 128 && v.size(3) <= 128, "CUDA v1 requires Q/K/V head dimensions <= 128");
    TORCH_CHECK(q.numel() && k.numel() && v.numel(), "Empty inputs are unsupported");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0 && std::isfinite(static_cast<float>(scale)), "scale must be positive finite FP32");
    block_size(bq, "q_block");
    block_size(bk, "kv_block");
    TORCH_CHECK(k_quant >= bk && k_quant <= 4096 && k_quant % bk == 0,
                "k_quant_block must be a multiple of kv_block and <= 4096");
    TORCH_CHECK(mean_type >= 0 && mean_type <= 2, "Invalid mean dtype");

    const auto original_type = q.scalar_type();
    auto transformed = preprocess_qk(q, k, k_smooth, hadamard);
    q = transformed[0]; k = transformed[1];
    v = v.to(at::kFloat).contiguous();
    vc::Shape s{};
    s.heads = small_int(q.size(0) * q.size(1), "B*H");
    s.nq = small_int(q.size(2), "Q sequence length");
    s.nk = small_int(k.size(2), "KV sequence length");
    s.d = small_int(q.size(3), "head dimension");
    s.dv = small_int(v.size(3), "V head dimension");
    s.bq = static_cast<int>(bq); s.bk = static_cast<int>(bk); s.k_quant = static_cast<int>(k_quant);
    s.nq_tiles = small_int((int64_t(s.nq) + bq - 1) / bq, "Q tile count");
    s.nk_tiles = small_int((int64_t(s.nk) + bk - 1) / bk, "KV tile count");
    s.nk_quant_tiles = small_int((int64_t(s.nk) + k_quant - 1) / k_quant, "K quantization tile count");
    s.pq = small_int(int64_t(s.nq_tiles) * bq, "padded Q length");
    s.pk = small_int(int64_t(s.nk_quant_tiles) * k_quant, "padded K length");
    s.dp = (s.d + 31) / 32 * 32; s.vp = (s.dv + 31) / 32 * 32;
    small_int(int64_t(s.heads) * s.nq_tiles, "Q preprocessing grid");
    small_int(int64_t(s.heads) * s.nk_quant_tiles, "K preprocessing grid");
    small_int(int64_t(s.heads) * s.nk_tiles * s.vp, "V preprocessing grid");
    s.smooth = smooth; s.expcast = expcast; s.mean_type = static_cast<int>(mean_type); s.scale = static_cast<float>(scale);

    // Every temporary is allocated on the current PyTorch stream, and all
    // kernels/cuBLASLt calls use that same stream. Allocator reuse is ordered.
    const auto f = q.options();
    const auto bytes = f.dtype(at::kByte);
    auto q8 = at::empty({s.heads, s.pq, s.dp}, bytes);
    auto k8 = at::empty({s.heads, s.pk, s.dp}, bytes);
    auto v8 = at::empty({s.heads, s.nk_tiles, s.vp, s.bk}, bytes);
    auto p8 = at::empty({s.bq, s.bk}, bytes);
    auto qs = at::empty({s.heads, s.nq_tiles}, f);
    auto ks = at::empty({s.heads, s.nk_quant_tiles}, f);
    auto vs = at::empty({s.heads, s.nk_tiles, s.vp}, f);
    const auto mean_dtype = mean_type == 1 ? at::kHalf : mean_type == 2 ? at::kBFloat16 : at::kFloat;
    auto means = at::empty(vs.sizes(), f.dtype(mean_dtype));
    auto scores = at::empty({s.bk, s.bq}, f);
    auto pv = at::empty({s.vp, s.bq}, f);
    auto maximum = at::empty({s.bq}, f);
    auto denom = at::empty_like(maximum);
    auto alpha = at::empty_like(maximum);
    auto mass = at::empty_like(maximum);
    auto accum = at::empty({s.bq, s.vp}, f);
    auto output = at::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, f);
    auto workspace = at::empty({32 * 1024 * 1024}, bytes);
    vc::Buffers buffers{
        q8.data_ptr<uint8_t>(), k8.data_ptr<uint8_t>(), v8.data_ptr<uint8_t>(), p8.data_ptr<uint8_t>(),
        qs.data_ptr<float>(), ks.data_ptr<float>(), vs.data_ptr<float>(), means.data_ptr(),
        scores.data_ptr<float>(), pv.data_ptr<float>(), maximum.data_ptr<float>(), denom.data_ptr<float>(),
        alpha.data_ptr<float>(), mass.data_ptr<float>(), accum.data_ptr<float>(), output.data_ptr<float>(),
        workspace.data_ptr(), static_cast<size_t>(workspace.numel())
    };
    auto handle = blaslt_handle(q.get_device());
    vc::forward(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), s, buffers,
                handle, c10::cuda::getCurrentCUDAStream(q.get_device()).stream());
    return output.to(original_type);
}

// The small Lloyd loop is host C++; ATen executes its arithmetic and stable
// sort on CUDA. No CPU copies and no Python iteration over heads/tokens.
std::vector<Tensor> group_values(Tensor v, int64_t clusters, int64_t iterations,
                                c10::optional<Tensor> old_centers) {
    require_hopper(v);
    const c10::cuda::CUDAGuard guard(v.device());
    const at::NoGradGuard no_grad;
    TORCH_CHECK(v.dim() == 4 && v.numel() > 0, "group_values expects nonempty [B,H,N,D]");
    TORCH_CHECK(clusters > 0 && iterations > 0, "clusters and iterations must be positive");
    auto x = v.to(at::kFloat);
    const int64_t count = std::min(clusters, x.size(2));
    Tensor centers;
    if (old_centers.has_value()) {
        centers = old_centers.value();
        TORCH_CHECK(centers.device() == x.device() && centers.scalar_type() == at::kFloat &&
                    centers.dim() == 4 && centers.size(0) == x.size(0) && centers.size(1) == x.size(1) &&
                    centers.size(2) == count && centers.size(3) == x.size(3), "Invalid cached centers");
    } else {
        auto indices = at::linspace(0, x.size(2) - 1, count, x.options()).to(at::kLong);
        centers = x.index_select(2, indices).clone();
    }
    auto labels = at::empty({x.size(0), x.size(1), x.size(2)}, x.options().dtype(at::kLong));
    for (int64_t iter = 0; iter < iterations; ++iter) {
        auto sums = at::zeros_like(centers);
        auto counts = at::zeros({x.size(0), x.size(1), count, 1}, x.options());
        auto center_norms = centers.square().sum(-1).unsqueeze(-2);
        for (int64_t start = 0; start < x.size(2); start += 2048) {
            auto rows = x.slice(2, start, std::min(start + 2048, x.size(2)));
            auto distances = rows.square().sum(-1, true) + center_norms - 2 * at::matmul(rows, centers.transpose(-2, -1));
            auto ids = distances.argmin(-1);
            labels.slice(2, start, start + rows.size(2)).copy_(ids);
            sums.scatter_add_(2, ids.unsqueeze(-1).expand_as(rows), rows);
            counts.scatter_add_(2, ids.unsqueeze(-1), at::ones_like(rows.slice(3, 0, 1)));
        }
        centers = at::where(counts > 0, sums / counts.clamp_min(1), centers);
    }
    return {at::argsort(labels, true, -1, false), centers};
}

Tensor expcast_codes(Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat, "ExpCast expects CUDA FP32 input");
    const c10::cuda::CUDAGuard guard(input.device());
    auto x = input.contiguous();
    auto result = at::empty(x.sizes(), x.options().dtype(at::kByte));
    vc::expcast(x.data_ptr<float>(), result.data_ptr<uint8_t>(), x.numel(),
                c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("numerical_contract_version") = 2;
    m.def("forward", &forward, "VC FP8 attention on Hopper (forward only)");
    m.def("preprocess_qk", &preprocess_qk, "Post-RoPE K centering and normalized Q/K Hadamard (CUDA FP32)");
    m.def("group_values", &group_values, "CUDA Lloyd clustering with stable permutation");
    m.def("expcast_codes", &expcast_codes, "ExpCast FP8 payload bytes");
    m.def("build_info", []() {
        pybind11::dict info;
        info["cuda_headers"] = CUDART_VERSION;
        info["cublaslt_version"] = cublasLtGetVersion();
        info["matmul"] = "cuBLASLt E4M3 x E4M3, FP32 output/accumulation";
        info["fully_fused_attention"] = false;
        info["numerical_contract_version"] = 2;
        info["expcast_rounding"] = "FP32 FMA then RNE";
        info["value_mean_storage"] = "mu / per-channel V scale at configured mean dtype";
        return info;
    });
}
