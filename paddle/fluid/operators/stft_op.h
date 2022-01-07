// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"

#include "paddle/fluid/operators/frame_op.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

enum class FFTNormMode : int64_t {
  none,       // No normalization
  by_sqrt_n,  // Divide by sqrt(signal_size)
  by_n,       // Divide by signal_size
};

// Enum representing the FFT type
enum class FFTTransformType : int64_t {
  C2C = 0,  // Complex-to-complex
  R2C,      // Real-to-complex
  C2R,      // Complex-to-real
};

// Returns true if the transform type has complex input
inline bool has_complex_input(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
    case FFTTransformType::C2R:
      return true;

    case FFTTransformType::R2C:
      return false;
  }
  PADDLE_THROW(platform::errors::InvalidArgument("Unknown FFTTransformType"));
}

// Returns true if the transform type has complex output
inline bool has_complex_output(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
    case FFTTransformType::R2C:
      return true;

    case FFTTransformType::C2R:
      return false;
  }
  PADDLE_THROW(platform::errors::InvalidArgument("Unknown FFTTransformType"));
}

namespace {
FFTNormMode get_norm_from_string(const std::string& norm, bool forward) {
  if (norm.empty() || norm == "backward") {
    return forward ? FFTNormMode::none : FFTNormMode::by_n;
  }

  if (norm == "forward") {
    return forward ? FFTNormMode::by_n : FFTNormMode::none;
  }

  if (norm == "ortho") {
    return FFTNormMode::by_sqrt_n;
  }

  PADDLE_THROW(platform::errors::InvalidArgument(
      "FFT norm string must be 'forward' or 'backward' or 'ortho', "
      "received %s",
      norm));
}
}  // anonymous namespace

template <typename T>
T compute_factor(int64_t size, FFTNormMode normalization) {
  constexpr auto one = static_cast<T>(1);
  switch (normalization) {
    case FFTNormMode::none:
      return one;
    case FFTNormMode::by_n:
      return one / static_cast<T>(size);
    case FFTNormMode::by_sqrt_n:
      return one / std::sqrt(static_cast<T>(size));
  }
  PADDLE_THROW(
      platform::errors::InvalidArgument("Unsupported normalization type"));
}

template <typename DeviceContext, typename Ti, typename To>
struct FFTC2CFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward);
};

template <typename DeviceContext, typename Ti, typename To>
struct FFTR2CFunctor {
  void operator()(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                  const std::vector<int64_t>& axes, FFTNormMode normalization,
                  bool forward);
};

template <typename DeviceContext, typename T>
class StftKernel : public framework::OpKernel<T> {
 public:
  /*
    Batch Signals (N, T) -> Frames (N, n_fft, num_frames) -> FFTR2C -> (N,
    n_fft/2 + 1, num_frames)
  */
  void Compute(const framework::ExecutionContext& ctx) const override {
    using C = paddle::platform::complex<T>;
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<C>(ctx.GetPlace());

    const size_t x_rank = x->dims().size();
    const size_t out_rank = out->dims().size();

    const int n_fft = ctx.Attr<int>("n_fft");
    const int hop_length = ctx.Attr<int>("hop_length");
    const std::string& norm_str = ctx.Attr<std::string>("normalization");
    auto axes = ctx.Attr<std::vector<int64_t>>(
        "axes");  // Pass from python list: [axis]

    const int n_frames = out->dims()[out_rank - 1];
    const int seq_length = x->dims()[x_rank - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // Frame
    Tensor frames;
    framework::DDim frames_dims(out->dims());
    frames_dims.at(axes[0]) = n_fft;
    frames.mutable_data<T>(frames_dims, ctx.GetPlace());

    FrameFunctor<DeviceContext, T>()(dev_ctx, x, &frames, seq_length, n_fft,
                                     n_frames, hop_length,
                                     /*is_grad*/ false);

    // FFTR2C with onesided
    auto normalization = get_norm_from_string(norm_str, true);
    FFTR2CFunctor<DeviceContext, T, C> fft_r2c_func;
    fft_r2c_func(dev_ctx, &frames, out, axes, normalization, true);
  }
};

}  // namespace operators
}  // namespace paddle
