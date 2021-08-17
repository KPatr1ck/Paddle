/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <cufft.h>
#include <cufftXt.h>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/spectral_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {

namespace {

using ScalarType = framework::proto::VarType::Type;
const int64_t kMaxCUFFTNdim = 3;
const int64_t kMaxDataNdim = kMaxCUFFTNdim + 1;

static inline std::string get_cufft_error_info(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
#ifndef __HIPCC__
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
#endif
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(cufftResult error) {
  if (error != CUFFT_SUCCESS) {
    std::ostringstream ss;
    ss << "cuFFT error: " << get_cufft_error_info(error);
    PADDLE_THROW(platform::errors::Fatal(ss.str()));
  }
}

// This struct is used to let us easily compute hashes of the
// parameters.
// It will be the **key** to the plan cache.
struct PlanKey {
  int64_t signal_ndim_;  // between 1 and kMaxCUFFTNdim, i.e., 1 <= signal_ndim
                         // <= 3
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxDataNdim];
  int64_t input_shape_[kMaxDataNdim];
  int64_t output_shape_[kMaxDataNdim];
  FFTTransformType fft_type_;
  ScalarType value_type_;

  PlanKey() = default;

  PlanKey(const std::vector<int64_t>& in_shape,
          const std::vector<int64_t>& out_shape,
          const std::vector<int64_t>& signal_size, FFTTransformType fft_type,
          ScalarType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_size.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    std::copy(signal_size.cbegin(), signal_size.cend(), sizes_);
    std::copy(in_shape.cbegin(), in_shape.cend(), input_shape_);
    std::copy(out_shape.cbegin(), out_shape.cend(), output_shape_);
  }
};

class CuFFTHandle {
  ::cufftHandle handle_;

 public:
  CuFFTHandle() { CUFFT_CHECK(cufftCreate(&handle_)); }

  ::cufftHandle& get() { return handle_; }
  const ::cufftHandle& get() const { return handle_; }

  ~CuFFTHandle() {
// Not using fftDestroy() for rocFFT to work around double freeing of handles
#ifndef __HIPCC__
    cufftDestroy(handle_);
#endif
  }
};

#ifdef __HIPCC__
using plan_size_type = int;
#else
using plan_size_type = long long int;  // NOLINT
#endif

// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   //2. whether to clone input before executing the plan
//   2. the workspace size needed
//
// This class will be the **value** in the plan cache.
// It **owns** the raw plan via a unique_ptr.
class CuFFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  CuFFTConfig(const CuFFTConfig&) = delete;
  CuFFTConfig& operator=(CuFFTConfig const&) = delete;

  explicit CuFFTConfig(const PlanKey& params)
      : CuFFTConfig(std::vector<int64_t>(
                        params.sizes_, params.sizes_ + params.signal_ndim_ + 1),
                    params.signal_ndim_, params.fft_type_, params.value_type_) {
  }

  // sizes are full signal, including batch size and always two-sided
  CuFFTConfig(std::vector<int64_t> sizes, const int64_t signal_ndim,
              FFTTransformType fft_type, ScalarType dtype)
      : fft_type_(fft_type), value_type_(dtype) {
    // signal sizes (excluding batch dim)
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const auto batch = static_cast<plan_size_type>(sizes[0]);
    // const int64_t signal_ndim = sizes.size() - 1;
    PADDLE_ENFORCE_EQ(signal_ndim, sizes.size() - 1,
                      platform::errors::InvalidArgument(
                          "The signal_ndim must be equal to sizes.size() - 1,"
                          "But signal_ndim is: [%d], sizes.size() - 1 is: [%d]",
                          signal_ndim, sizes.size() - 1));

// Since cuFFT has limited non-unit stride support and various constraints, we
// use a flag to keep track throughout this function to see if we need to
// input = input.clone();
/*
#ifdef __HIPCC__
    // clone input to avoid issues with hipfft clobering the input and failing
tests
    clone_input = true;
#else
    clone_input = false;
#endif

    CuFFTDataLayout in_layout;
    if (clone_input) {
      in_layout = cufft_simple_embed(sizes, fft_type == FFTTransformType::C2R);
    } else {
      in_layout = as_cufft_embed(in_strides, sizes, fft_type ==
FFTTransformType::C2R);
    }
    auto out_layout = as_cufft_embed(out_strides, sizes, fft_type ==
FFTTransformType::R2C);
    TORCH_INTERNAL_ASSERT(!out_layout.must_clone, "Out strides cannot be
represented as CuFFT embedding");
    clone_input |= in_layout.must_clone;

    // Check if we can take advantage of simple data layout.
    //
    // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.

    const bool simple_layout = in_layout.simple && out_layout.simple;
*/
#ifdef __HIPCC__
    hipfftType exec_type = [&] {
      if (dtype == framework::proto::VarType::FP32) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_C2C;
          case FFTTransformType::R2C:
            return HIPFFT_R2C;
          case FFTTransformType::C2R:
            return HIPFFT_C2R;
        }
      } else if (dtype == framework::proto::VarType::FP64) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_Z2Z;
          case FFTTransformType::R2C:
            return HIPFFT_D2Z;
          case FFTTransformType::C2R:
            return HIPFFT_Z2D;
        }
      }
      PADDLE_THROW(platform::errors::InvalidArgument(
          "hipFFT only support transforms of type float32 and float64"));
    }();
#else
    cudaDataType itype, otype, exec_type;
    const auto complex_input = has_complex_input(fft_type);
    const auto complex_output = has_complex_output(fft_type);
    if (dtype == framework::proto::VarType::FP32) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (dtype == framework::proto::VarType::FP64) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (dtype == framework::proto::VarType::FP16) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "cuFFT doesn't support tensor of type: [%s]", dtype));
    }
#endif

    // disable auto allocation of workspace to use THC allocator
    CUFFT_CHECK(cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

// make plan
/*
    if (simple_layout) {
      // If with unit-stride, we tell cuFFT by setting inembed == onembed ==
   NULL.
      // In such case, cuFFT ignores istride, ostride, idist, and odist
      // by assuming istride = ostride = 1.
      //
      // See NOTE [ cuFFT Embedded Strides ] in native/cuda/SpectralOps.cu.
*/
#ifdef __HIPCC__
    CUFFT_CHECK(hipfftMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, exec_type,
        batch, &ws_size_t));
#else
    CUFFT_CHECK(cufftXtMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));
#endif
    /*
        } else {
    #ifdef __HIPCC__
          CUFFT_CHECK(hipfftMakePlanMany(plan(), signal_ndim,
    signal_sizes.data(),
            in_layout.embed.data(), in_layout.stride, in_layout.dist,
            out_layout.embed.data(), out_layout.stride, out_layout.dist,
            exec_type, batch, &ws_size_t));
    #else
          CUFFT_CHECK(cufftXtMakePlanMany(plan(), signal_ndim,
    signal_sizes.data(),
                in_layout.embed.data(), in_layout.stride, in_layout.dist, itype,
                out_layout.embed.data(), out_layout.stride, out_layout.dist,
    otype,
                batch, &ws_size_t, exec_type));
    #endif
        }
    */
    ws_size = ws_size_t;
  }

  const cufftHandle& plan() const { return plan_ptr.get(); }

  FFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  size_t workspace_size() const { return ws_size; }

 private:
  CuFFTHandle plan_ptr;
  size_t ws_size;
  FFTTransformType fft_type_;
  ScalarType value_type_;
};

// Hashing machinery for Key
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Key>
struct KeyHash {
  // Key must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  size_t operator()(const Key& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < static_cast<int>(sizeof(Key)); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Key>
struct KeyEqual {
  // Key must be a POD because we read out its memory
  // contenst as char* when comparing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  bool operator()(const Key& a, const Key& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Key)) == 0;
  }
};

#if CUDA_VERSION < 10000
// Note that the max plan number for CUDA version < 10 has to be 1023
// due to a bug that fails on the 1024th plan
constexpr size_t CUFFT_MAX_PLAN_NUM = 1023;
constexpr size_t CUFFT_DEFAULT_CACHE_SIZE = CUFFT_MAX_PLAN_NUM;
#else
constexpr size_t CUFFT_MAX_PLAN_NUM = std::numeric_limits<size_t>::max();
// The default max cache size chosen for CUDA version > 10 is arbitrary.
// This number puts a limit on how big of a plan cache should we maintain by
// default. Users can always configure it via cufft_set_plan_cache_max_size.
constexpr size_t CUFFT_DEFAULT_CACHE_SIZE = 4096;
#endif
static_assert(CUFFT_MAX_PLAN_NUM >= 0 &&
                  CUFFT_MAX_PLAN_NUM <= std::numeric_limits<size_t>::max(),
              "CUFFT_MAX_PLAN_NUM not in size_t range");
static_assert(CUFFT_DEFAULT_CACHE_SIZE >= 0 &&
                  CUFFT_DEFAULT_CACHE_SIZE <= CUFFT_MAX_PLAN_NUM,
              "CUFFT_DEFAULT_CACHE_SIZE not in [0, CUFFT_MAX_PLAN_NUM] range");

// This cache assumes that the mapping from key to value never changes.
// This is **NOT** thread-safe. Please use a mutex when using it **AND** the
// value returned from try_emplace_value.
// The contract of using this cache is that try_emplace_value should only be
// used when the max_size is positive.
class PlanLRUCache {
 public:
  using kv_t = typename std::pair<PlanKey, CuFFTConfig>;
  using map_t =
      typename std::unordered_map<std::reference_wrapper<PlanKey>,
                                  typename std::list<kv_t>::iterator,
                                  KeyHash<PlanKey>, KeyEqual<PlanKey>>;
  using map_kkv_iter_t = typename map_t::iterator;

  PlanLRUCache() : PlanLRUCache(CUFFT_DEFAULT_CACHE_SIZE) {}

  explicit PlanLRUCache(int64_t max_size) { _set_max_size(max_size); }

  PlanLRUCache(PlanLRUCache&& other) noexcept
      : _usage_list(std::move(other._usage_list)),
        _cache_map(std::move(other._cache_map)),
        _max_size(other._max_size) {}

  PlanLRUCache& operator=(PlanLRUCache&& other) noexcept {
    _usage_list = std::move(other._usage_list);
    _cache_map = std::move(other._cache_map);
    _max_size = other._max_size;
    return *this;
  }

  // If key is in this cache, return the cached config. Otherwise, emplace the
  // config in this cache and return it.
  // Return const reference because CuFFTConfig shouldn't be tampered with once
  // created.
  const CuFFTConfig& lookup(PlanKey params) {
    PADDLE_ENFORCE_GT(_max_size, 0,
                      platform::errors::InvalidArgument(
                          "The max size of PlanLRUCache must be great than 0,"
                          "But received is [%d]",
                          _max_size));

    map_kkv_iter_t map_it = _cache_map.find(params);
    // Hit, put to list front
    if (map_it != _cache_map.end()) {
      _usage_list.splice(_usage_list.begin(), _usage_list, map_it->second);
      return map_it->second->second;
    }

    // Miss
    // remove if needed
    if (_usage_list.size() >= _max_size) {
      auto last = _usage_list.end();
      last--;
      _cache_map.erase(last->first);
      _usage_list.pop_back();
    }

    // construct new plan at list front, then insert into _cache_map
    _usage_list.emplace_front(std::piecewise_construct,
                              std::forward_as_tuple(params),
                              std::forward_as_tuple(params));
    auto kv_it = _usage_list.begin();
    _cache_map.emplace(std::piecewise_construct,
                       std::forward_as_tuple(kv_it->first),
                       std::forward_as_tuple(kv_it));
    return kv_it->second;
  }

  void clear() {
    _cache_map.clear();
    _usage_list.clear();
  }

  void resize(int64_t new_size) {
    _set_max_size(new_size);
    auto cur_size = _usage_list.size();
    if (cur_size > _max_size) {
      auto delete_it = _usage_list.end();
      for (size_t i = 0; i < cur_size - _max_size; i++) {
        delete_it--;
        _cache_map.erase(delete_it->first);
      }
      _usage_list.erase(delete_it, _usage_list.end());
    }
  }

  size_t size() const { return _cache_map.size(); }

  size_t max_size() const noexcept { return _max_size; }

  std::mutex mutex;

 private:
  // Only sets size and does value check. Does not resize the data structures.
  void _set_max_size(int64_t new_size) {
    // We check that 0 <= new_size <= CUFFT_MAX_PLAN_NUM here. Since
    // CUFFT_MAX_PLAN_NUM is of type size_t, we need to do non-negativity check
    // first.
    PADDLE_ENFORCE_GE(
        new_size, 0,
        platform::errors::InvalidArgument(
            "cuFFT plan cache size must be non-negative, But received is [%d]",
            new_size));
    PADDLE_ENFORCE_LE(new_size, CUFFT_MAX_PLAN_NUM,
                      platform::errors::InvalidArgument(
                          "cuFFT plan cache size can not be larger than [%d], "
                          "But received is [%d]",
                          CUFFT_MAX_PLAN_NUM, new_size));
    _max_size = static_cast<size_t>(new_size);
  }

  std::list<kv_t> _usage_list;
  map_t _cache_map;
  size_t _max_size;
};

// Execute a pre-planned transform
static void exec_cufft_plan(const CuFFTConfig& config, void* in_data,
                            void* out_data, bool forward) {
  auto& plan = config.plan();
#ifdef __HIPCC__
  auto value_type = config.data_type();
  if (value_type == framework::proto::VarType::FP32) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        CUFFT_CHECK(hipfftExecC2C(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftComplex*>(out_data),
                                  forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        CUFFT_CHECK(hipfftExecR2C(plan, static_cast<hipfftReal*>(in_data),
                                  static_cast<hipfftComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecC2R(plan, static_cast<hipfftComplex*>(in_data),
                                  static_cast<hipfftReal*>(out_data)));
        return;
      }
    }
  } else if (value_type == framework::proto::VarType::FP64) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        CUFFT_CHECK(hipfftExecZ2Z(plan,
                                  static_cast<hipfftDoubleComplex*>(in_data),
                                  static_cast<hipfftDoubleComplex*>(out_data),
                                  forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        CUFFT_CHECK(hipfftExecD2Z(plan, static_cast<hipfftDoubleReal*>(in_data),
                                  static_cast<hipfftDoubleComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        CUFFT_CHECK(hipfftExecZ2D(plan,
                                  static_cast<hipfftDoubleComplex*>(in_data),
                                  static_cast<hipfftDoubleReal*>(out_data)));
        return;
      }
    }
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "hipFFT doesn't support transforms on type: [%s]", value_type));
#else
  CUFFT_CHECK(cufftXtExec(plan, in_data, out_data,
                          forward ? CUFFT_FORWARD : CUFFT_INVERSE));
#endif
}

static std::vector<std::unique_ptr<PlanLRUCache>> plan_caches;
static std::mutex plan_caches_mutex;

static inline PlanLRUCache& cufft_get_plan_cache(int64_t device_index) {
  std::lock_guard<std::mutex> guard(plan_caches_mutex);

  /*
  PADDLE_ENFORCE_GE(device_index, static_cast<int64_t>(0),
                    platform::errors::InvalidArgument(
                        "cuFFT device index must be greater than or equal to "
                        "0, But received is [%d]" device_index));
  */

  if (device_index >= plan_caches.size()) {
    plan_caches.resize(device_index + 1);
  }

  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<PlanLRUCache>();
  }

  return *plan_caches[device_index];
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
template <typename DeviceContext, typename T>
void exec_fft(const DeviceContext& ctx, Tensor* out, const Tensor* X,
              const std::vector<int64_t>& out_sizes,
              const std::vector<int64_t>& dim, bool forward) {
  const auto x_dims = X->dim() const auto ndim =
      static_cast<int64_t>(X->dim().size());
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Transpose batch dimensions first, then with transforming dims
  std::vector<int> dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int{0});
  std::vector<bool> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(),
                     [&](int64_t d) { return !is_transformed_dim[d]; });
  std::sort(dim_permute.begin(), batch_end);
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  framework::DDim trans_dims(X->dim());
  for (size_t i = 0; i < ndim; i++) {
    trans_dims[i] = x_dims[dim_permute[i]];   // shape of input transpose
    reverse_dim_permute[dim_permute[i]] = i;  // reverse of dim permute
  }
  framework::Tensor input;
  input.Resize(trans_dims) input.mutable_data<T>(ctx.GetPlace());
  auto ret = TransposeSimple<T>::run(ctx, *X, dim_permute, input);
  if (!ret) {
    TransCompute<DeviceContext, T>(ndim, ctx, *X, input, dim_permute);
  }

  // Reshape batch dimensions into a single dimension
  std::vector<int64_t> batched_sizes(signal_ndim + 1);
  auto batch_size =
      std::accumulate(dim_permute.begin(), batch_end, static_cast<int>(1),
                      std::multiplies<int>());
  batched_sizes[0] = batch_size;
  std::copy(dim.cbegin(), dim.cend(), batched_sizes.begin() + 1);
  input->Resize(batched_sizes);

  // Check the shape of transforming dims with input and output
  std::vector<int64_t> signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input->dims()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    PADDLE_ENFORCE_EQ(
        (in_size == signal_size[i + 1] ||
         in_size == (signal_size[i + 1] / 2) + 1),
        true,
        platform::errors::InvalidArgument(
            "The dimension[%d] of Input size: [%d] must be equal or half to "
            "The dimension[%d] of Output size: [%d]"
            "Input(Scale) is [%d]",
            dim[i], in_size, dim[i], out_size));
    PADDLE_ENFORCE_EQ(
        (out_size == signal_size[i + 1] ||
         out_size == (signal_size[i + 1] / 2) + 1),
        true,
        platform::errors::InvalidArgument(
            "The dimension[%d] of Output size: [%d] must be equal or half to "
            "The dimension[%d] of Input size: [%d]"
            "Input(Scale) is [%d]",
            dim[i], out_size, dim[i], in_size));
  }

  std::vector<int64_t> reshape_out_sizes(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    reshape_out_sizes[i] = out_sizes[dim_permute[i]];
  }
  std::vector<int64_t> batched_out_sizes(batched_sizes.begin(),
                                         batched_sizes.end());
  for (size_t i = 0; i < dim.size(); ++i) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }

  // output
  framework::Tensor output;
  output->Resize(batched_out_sizes) output.mutable_data<T>(ctx.GetPlace());

  // Create the transform plan (either from cache or locally)
  const auto value_type = framework::ToRealType(input.type());
  auto fft_type = GetFFTTransformType(input.type(), output.type());
  PlanKey Key(framework::vectorize(input->dim()),
              framework::vectorize(output->dim()), signal_size, fft_type,
              value_type);
  PlanLRUCache& plan_cache = cufft_get_plan_cache(input.device().index());
  std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
  c10::optional<CuFFTConfig> uncached_plan;
  const CuFFTConfig* config = nullptr;

  if (plan_cache.max_size() > 0) {
    guard.lock();
    if (plan_cache.max_size() > 0) {  // check again after acquiring the lock
      config = &plan_cache.lookup(Key);
    }
  }

  if (config == nullptr) {
    uncached_plan.emplace(Key);
    config = &uncached_plan.value();
  }

  auto& plan = config->plan();

  // prepare cufft for execution
  // CUFFT_CHECK(cufftSetStream(plan, reinterpret_cast<const
  // platform::CUDADeviceContext&>(ctx).stream()));
  framework::Tensor workspace_tensor;
  workspace_tensor.mutable_data<T>(ctx.GetPlace(),
                                   requested_size = config->workspace_size());
  CUFFT_CHECK(cufftSetWorkArea(plan, workspace.data()));

  // execute transform plan
  exec_cufft_plan(*config, input.data_ptr(), output.data_ptr(), forward);

  // Inverting output by reshape and transpose to original batch and dimension
  output->Resize(reshape_out_sizes);
  // Todo: transpose out
  out->Resize(out_sizes) auto ret =
      TransposeSimple<T>::run(ctx, *output, reverse_dim_permute, out);
  if (!ret) {
    TransCompute<DeviceContext, T>(ndim, ctx, *output, out,
                                   reverse_dim_permute);
  }

  /*
  std::vector<int64_t> out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (int64_t i = batch_dims; i < ndim; ++i) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  return out.as_strided_(out_sizes, out_strides, out.storage_offset());
  */
}

// Calculates the normalization constant and applies it in-place to out
// sizes is the sizes of a twosided tensor and dims are all transformed dims
double fft_normalization_scale(FFTNormMode normalization,
                               const std::vector<int64_t>& sizes,
                               const std::vector<int64_t>& dims) {
  // auto norm = static_cast<fft_norm_mode>(normalization);
  if (normalization == FFTNormMode::none) {
    return 1.0;
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (normalization == FFTNormMode::by_sqrt_n)
                                 ? std::sqrt(signal_numel)
                                 : static_cast<double>(signal_numel);
  return 1.0 / scale_denom;
}

void exec_normalization(Tensor* out, FFTNormMode normalization,
                        const std::vector<int64_t>& sizes,
                        const std::vector<int64_t>& axes) {
  auto scale = fft_normalization_scale(normalization, sizes, axes);
  if (scale != 1.0) {
    out->mul(scale);
  }
}

/*
template <typename DeviceContext, typename T>
void fft_c2c_cufft(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                   const std::vector<int64_t>& axes, FFTNormMode normalization,
                   bool forward) {
  if (axes.empty()) {
    framework::TensorCopy(*X, ctx.GetPlace(), out);
    return;
  }

  auto out_dims = framework::vectorize(X->dims());
  std::vector<int64_t> working_axes(axes.begin(), axes.end());
  framework::Tensor working_tensor;
  working_tensor.mutable_data<T>(ctx.GetPlace());
  framework::TensorCopy(*X, ctx.GetPlace(), &working_tensor);

  while (true) {
    const auto max_dims =
        std::min(static_cast<size_t>(kMaxCUFFTNdim), working_axes.size());
    auto first_dims =
        std::vector<int64_t>(working_axes.end() - max_dims, working_axes.end());

    exec_fft<DeviceContext, T>(ctx, out, working_tensor, out_dims, first_dims,
                               forward);
    working_axes.resize(working_axes.size() - max_dims);

    if (working_axes.empty()) {
      break;
    }

    std::swap(*out, working_tensor);
  }

  exec_normalization(output, normalization, out_dims, dim);
}

template <typename DeviceContext, typename T>
void fft_c2c_cufft_backward(const DeviceContext& ctx, const Tensor* d_y,
                            Tensor* d_x, const std::vector<int64_t>& axes,
                            FFTNormMode normalization, bool forward) {
  fft_c2c_cufft(ctx, d_y, d_x, axes, normalization, forward);
}
*/

}  // anonymous namespace

template <typename T>
struct FFTC2CFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx, const Tensor* X,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    if (axes.empty()) {
      framework::TensorCopy(*X, ctx.GetPlace(), out);
      return;
    }

    auto out_dims = framework::vectorize(X->dims());
    std::vector<int64_t> working_axes(axes.begin(), axes.end());
    framework::Tensor working_tensor;
    working_tensor.mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(*X, ctx.GetPlace(), &working_tensor);

    while (true) {
      const auto max_dims =
          std::min(static_cast<size_t>(kMaxCUFFTNdim), working_axes.size());
      auto first_dims = std::vector<int64_t>(working_axes.end() - max_dims,
                                             working_axes.end());

      exec_fft<CUDADeviceContext, T>(ctx, out, working_tensor, out_dims,
                                     first_dims, forward);
      working_axes.resize(working_axes.size() - max_dims);

      if (working_axes.empty()) {
        break;
      }

      std::swap(*out, working_tensor);
    }
    exec_normalization(out, normalization, out_dims, axes);
  }
};

template <typename T>
class FFTC2CKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context();

    // axes must be sorted before Compute
    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    y->mutable_data<U>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2CFunctor<platform::CUDADeviceContext, U>(dev_ctx, x, y, axes,
                                                  normalization, forward);
  }
};

template <typename T>
class FFTC2CGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const int64_t normalization = ctx.Attr<int64_t>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Out"));

    d_y->mutable_data<T>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    FFTC2CFunctor<platform::CUDADeviceContext, U>(dev_ctx, d_y, d_x, axes,
                                                  normalization, forward);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, double>);
