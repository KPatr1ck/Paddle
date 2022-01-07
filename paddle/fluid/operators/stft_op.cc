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

#include "paddle/fluid/operators/stft_op.h"
#include "extern_pocketfft/pocketfft_hdronly.h"

namespace paddle {
namespace operators {

template <typename Ti, typename To>
struct FFTR2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* X,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = Ti;
    using C = std::complex<R>;

    const auto& input_dim = X->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<std::ptrdiff_t> in_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<std::ptrdiff_t> out_strides =
        framework::vectorize<std::ptrdiff_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [&](std::ptrdiff_t s) { return s * data_size; });
    }

    const auto* in_data = X->data<R>();
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // pocketfft requires std::vector<size_t>
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet normalization factor
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::r2c(in_sizes, in_strides, out_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

class StftOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "frame");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "frame");

    const int n_fft = ctx->Attrs().Get<int>("n_fft");
    const int hop_length = ctx->Attrs().Get<int>("hop_length");

    const auto x_dims = ctx->GetInputDim("X");
    const int x_rank = x_dims.size();

    PADDLE_ENFORCE_EQ(
        x_rank, 2,
        platform::errors::InvalidArgument(
            "Input(X) of StftOp should be a tensor with shape [N, T], "
            "but got rank %s.",
            x_rank));
    PADDLE_ENFORCE_GT(hop_length, 0,
                      platform::errors::InvalidArgument(
                          "Attribute(hop_length) of FrameOp should be greater "
                          "than 0, but got %s.",
                          hop_length));

    std::vector<int64_t> output_shape;
    int seq_length;
    int n_frames;

    seq_length = x_dims[x_rank - 1];

    PADDLE_ENFORCE_LE(n_fft, seq_length,
                      platform::errors::InvalidArgument(
                          "Attribute(frame_length) of FrameOp should be less "
                          "equal than sequence length, but got (%s) > (%s).",
                          n_fft, seq_length));

    output_shape.push_back(x_dims[0]);
    output_shape.push_back(n_fft / 2 + 1);
    n_frames = 1 + (seq_length - n_fft) / hop_length;
    output_shape.push_back(n_frames);

    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

class StftOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "");
    AddOutput("Out", "");
    AddAttr<int>("n_fft", "");
    AddAttr<int>("hop_length", "");
    AddAttr<std::string>("normalization", "");
    AddAttr<std::vector<int64_t>>("axes", "");
    AddComment(R"DOC(
      Stft Op.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(stft, ops::StftOp, ops::StftOpMaker);
REGISTER_OP_CPU_KERNEL(
    stft, ops::StftKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StftKernel<paddle::platform::CPUDeviceContext, double>);
