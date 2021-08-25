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

#include "paddle/fluid/operators/spectral_op.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/complex.h"

#if defined(PADDLE_WITH_ONEMKL)
#include <mkl_dfti.h>
// #include "mkl_service.h"
#elif defined(PADDLE_WITH_POCKETFFT)
#include "extern_pocketfft/pocketfft_hdronly.h"
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

//////////////// C2C
class FFTC2COpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_c2c op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_c2c op.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<std::string>("normalization",
                         "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddComment(R"DOC(
      // add doc here
    )DOC");
  }
};

class FFTC2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of FFTC2COp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(%s) of FFTC2COp should not be null.", "Out"));

    ctx->ShareDim("X", /*->*/ "Out");  // only for c2c
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FFTC2CGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_c2c_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTC2CGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::InvalidArgument(
            "Input(%s) of FFTC2CGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument(
            "Output(%s) of FFTC2CGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

///////////////// R2C
class FFTR2COpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_r2c op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_r2c op.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<std::string>("normalization",
                         "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddAttr<bool>("onesided", "bool, perform onesided fft.");
    AddComment(R"DOC(
      // add doc here
    )DOC");
  }
};

class FFTR2COp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of FFTC2ROp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(%s) of FFTC2ROp should not be null.", "Out"));
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    const bool onesided = ctx->Attrs().Get<bool>("onesided");
    if (!onesided) {
      ctx->ShareDim("X", /*->*/ "Out");  //
    } else {
      framework::DDim out_dim(ctx->GetInputDim("X"));
      const int64_t last_fft_axis = axes.back();
      const int64_t last_fft_dim_size = out_dim.at(last_fft_axis);
      out_dim.at(last_fft_axis) = last_fft_dim_size / 2 + 1;
      ctx->SetOutputDim("Out", out_dim);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FFTR2CGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_r2c_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTR2CGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::InvalidArgument(
            "Input(%s) of FFTR2CGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument(
            "Output(%s) of FFTR2CGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    auto out_grad_name = framework::GradVarName("Out");
    const bool onesided = ctx->Attrs().Get<bool>("onesided");
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");
    if (!onesided) {
      ctx->ShareDim(out_grad_name, /*->*/ x_grad_name);  //
    } else {
      const auto out_grad_dim = ctx->GetInputDim(out_grad_name);
      framework::DDim x_grad_dim(out_grad_dim);
      const int64_t last_fft_axis = axes.back();
      const int64_t last_fft_dim_size = x_grad_dim.at(last_fft_axis);
      x_grad_dim.at(last_fft_axis) = (last_fft_dim_size - 1) * 2;
      ctx->SetOutputDim(x_grad_name, x_grad_dim);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

//////////////// C2R
class FFTC2ROpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), the input tensor of fft_c2r op.");
    AddOutput("Out", "(Tensor), the output tensor of fft_c2r op.");
    AddAttr<std::vector<int64_t>>("axes",
                                  "std::vector<int64_t>, the fft axes.");
    AddAttr<std::string>("normalization",
                         "fft_norm_type, the fft normalization type.");
    AddAttr<bool>("forward", "bool, the fft direction.");
    AddComment(R"DOC(
      // add doc here
    )DOC");
  }
};

class FFTC2ROp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of FFTC2ROp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(%s) of FFTC2ROp should not be null.", "Out"));
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");

    framework::DDim out_dim(ctx->GetInputDim("X"));
    const int64_t last_fft_axis = axes.back();
    const int64_t last_fft_dim_size = out_dim.at(last_fft_axis);
    out_dim.at(last_fft_axis) = (last_fft_dim_size - 1) * 2;
    ctx->SetOutputDim("Out", out_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    const auto kernel_dtype = framework::ToRealType(in_dtype);
    return framework::OpKernelType(kernel_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FFTC2RGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fft_c2r_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class FFTC2RGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::InvalidArgument(
            "Input(%s) of FFTC2RGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument(
            "Output(%s) of FFTC2RGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    auto out_grad_name = framework::GradVarName("Out");
    const auto axes = ctx->Attrs().Get<std::vector<int64_t>>("axes");

    const auto out_grad_dim = ctx->GetInputDim(out_grad_name);
    framework::DDim x_grad_dim(out_grad_dim);
    const int64_t last_fft_axis = axes.back();
    const int64_t last_fft_dim_size = x_grad_dim.at(last_fft_axis);
    x_grad_dim.at(last_fft_axis) = last_fft_dim_size / 2 + 1;
    ctx->SetOutputDim(x_grad_name, x_grad_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

//////////////// common
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
      "Fft norm string must be forward or backward or ortho"));
}

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
  PADDLE_THROW("Unsupported normalization type");
}

////////////////// Functors
#if defined(PADDLE_WITH_ONEMKL)

static inline void MKL_DFTI_CHECK(MKL_INT status) {
  if (status && !DftiErrorClass(status, DFTI_NO_ERROR)) {
    PADDLE_THROW(DftiErrorMessage(status));
  }
}

struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR_HANDLE handle) {
    if (handle != nullptr) {
      MKL_DFTI_CHECK(DftiFreeDescriptor(&handle));
    }
  }
};

class DftiDescriptor {
 public:
  void init(DFTI_CONFIG_VALUE precision, DFTI_CONFIG_VALUE signal_type,
            MKL_LONG signal_ndim, MKL_LONG* sizes) {
    if (desc_ != nullptr) {
      PADDLE_THROW("DFT DESCRIPTOR can only be initialized once.");
    }
    DFTI_DESCRIPTOR* raw_desc;
    if (signal_ndim == 1) {
      MKL_DFTI_CHECK(
          DftiCreateDescriptor(&raw_desc, precision, signal_type, 1, sizes[0]));
    } else {
      MKL_DFTI_CHECK(
          DftiCreateDescriptor(&raw_desc, precision, signal_type, 1, sizes[0]));
    }
    desc_.reset(raw_desc);
  }

  DFTI_DESCRIPTOR* get() const {
    if (desc_ == nullptr) {
      PADDLE_THROW("DFTI DESCRIPTOR has not been initialized.");
    }
    return desc_.get();
  }

 private:
  std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter> desc_;
};

DftiDescriptor _plan_mkl_fft(const framework::proto::VarType::Type& in_dtype,
                             const framework::proto::VarType::Type& out_dtype,
                             const framework::DDim& in_strides,
                             const framework::DDim& out_strides,
                             const std::vector<int>& signal_sizes,
                             FFTNormMode normalization, bool forward) {
  const DFTI_CONFIG_VALUE precision = [&] {
    switch (in_dtype) {
      case framework::proto::VarType::FP32:
        return DFTI_SINGLE;
      case framework::proto::VarType::COMPLEX64:
        return DFTI_SINGLE;
      case framework::proto::VarType::FP64:
        return DFTI_DOUBLE;
      case framework::proto::VarType::COMPLEX128:
        return DFTI_SINGLE;
      default:
        PADDLE_THROW("MKL DFT does not support.");
    }
  }();

  const bool complex_input = framework::IsComplexType(in_dtype);
  const bool complex_output = framework::IsComplexType(out_dtype);
  const DFTI_CONFIG_VALUE domain = [&] {
    if (forward) {
      return complex_input ? DFTI_COMPLEX : DFTI_REAL;
    } else {
      return complex_output ? DFTI_COMPLEX : DFTI_REAL;
    }
  }();

  DftiDescriptor descriptor;  /////
  std::vector<MKL_LONG> fft_sizes(signal_sizes.cbegin(), signal_sizes.cend());
  const MKL_LONG signal_ndim = fft_sizes.size() - 1;
  descriptor.init(precision, domain, signal_ndim, fft_sizes.data() + 1);

  // placement inplace?
  MKL_DFTI_CHECK(
      DftiSetValue(descriptor.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE));

  // number of transformation
  const MKL_LONG batch_size = fft_sizes[0];
  MKL_DFTI_CHECK(
      DftiSetValue(descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, batch_size));

  // input & output distance
  const MKL_LONG idist = in_strides[0];
  const MKL_LONG odist = out_strides[0];
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_INPUT_DISTANCE, idist));
  MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_OUTPUT_DISTANCE, odist));

  // input & output stride
  std::vector<MKL_LONG> mkl_in_stride(1 + signal_ndim, 0);
  std::vector<MKL_LONG> mkl_out_stride(1 + signal_ndim, 0);
  for (MKL_LONG i = 1; i <= signal_ndim; i++) {
    mkl_in_stride[i] = in_strides[i];
    mkl_out_stride[i] = out_strides[i];
  }
  MKL_DFTI_CHECK(
      DftiSetValue(descriptor.get(), DFTI_INPUT_STRIDES, mkl_in_stride));
  MKL_DFTI_CHECK(
      DftiSetValue(descriptor.get(), DFTI_OUTPUT_STRIDES, mkl_out_stride));

  // conjugate even storage
  if (!complex_input || !complex_output) {
    MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), DFTI_CONJUGATE_EVEN_STORAGE,
                                DFTI_COMPLEX_COMPLEX));
  }

  MKL_LONG signal_numel =
      std::accumulate(fft_sizes.cbegin() + 1, fft_sizes.cend(), 1UL,
                      std::multiplies<MKL_LONG>());
  if (normalization != FFTNormMode::none) {
    const double scale =
        ((normalization == FFTNormMode::by_sqrt_n)
             ? 1.0 / std::sqrt(static_cast<double>(signal_numel))
             : 1.0 / static_cast<double>(signal_numel));
    const auto scale_direction =
        forward ? DFTI_FORWARD_SCALE : DFTI_BACKWARD_SCALE;
    MKL_DFTI_CHECK(DftiSetValue(descriptor.get(), scale_direction, scale));
  }

  // commit the descriptor
  MKL_DFTI_CHECK(DftiCommitDescriptor(descriptor.get()));
  return descriptor;
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
template <typename DeviceContext, typename Ti, typename To>
void exec_fft(const DeviceContext& ctx, const Tensor* x, Tensor* out,
              const std::vector<int64_t>& axes, FFTNormMode normalization,
              bool forward) {
  const framework::DDim& in_sizes = x->dims();
  const int ndim = in_sizes.size();
  const int signal_ndim = axes.size();
  const int batch_ndim = ndim - signal_ndim;
  const framework::DDim& out_sizes = out->dims();

  // make a dim permutation
  std::vector<int> dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::vector<bool> is_transformed_dim(ndim, false);
  for (const auto& d : axes) {
    is_transformed_dim[d] = true;
  }
  const auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(),
                     [&](size_t axis) { return !is_transformed_dim[axis]; });
  std::copy(axes.cbegin(), axes.cend(), batch_end);

  // transpose input according to that permutation
  framework::DDim transposed_input_shape = in_sizes.transpose(dim_permute);
  std::vector<int64_t> transposed_input_shape_ =
      framework::vectorize(transposed_input_shape);
  framework::Tensor transposed_input;
  transposed_input.Resize(transposed_input_shape);
  const auto place = ctx.GetPlace();
  transposed_input.mutable_data<Ti>(place);
  TransCompute<platform::CPUDeviceContext, Ti>(ndim, ctx, *x, &transposed_input,
                                               dim_permute);

  // make an collapsed input: collapse batch axes for input
  const int batch_size = std::accumulate(
      transposed_input_shape.Get(), transposed_input_shape.Get() + batch_ndim,
      1L, std::multiplies<int64_t>());
  std::vector<int> collapsed_input_shape_(1 + signal_ndim);
  collapsed_input_shape_[0] = batch_size;
  std::copy(transposed_input_shape_.begin() + batch_ndim,
            transposed_input_shape_.end(), collapsed_input_shape_.begin() + 1);
  const framework::DDim collapsed_input_shape =
      framework::make_ddim(collapsed_input_shape_);
  transposed_input.Resize(collapsed_input_shape);
  framework::Tensor& collapsed_input = transposed_input;

  // make a collapsed output
  std::vector<int> collapsed_output_shape_(1 + signal_ndim);
  collapsed_output_shape_[0] = batch_size;
  for (int i = 0; i < signal_ndim; i++) {
    collapsed_output_shape_[1 + i] = out_sizes[axes[i]];
  }
  const framework::DDim collapsed_output_shape =
      framework::make_ddim(collapsed_output_shape_);
  framework::Tensor collapsed_output;
  collapsed_output.Resize(collapsed_output_shape);
  collapsed_output.mutable_data(place, out->type());

  // signal sizes
  std::vector<int> signal_sizes(1 + signal_ndim);
  signal_sizes[0] = batch_size;
  for (int i = 0; i < signal_ndim; i++) {
    signal_sizes[1 + i] =
        std::max(collapsed_input_shape[1 + i], collapsed_output_shape[1 + i]);
  }

  // input & output stride
  const framework::DDim input_stride = framework::stride(collapsed_input_shape);
  const framework::DDim output_stride =
      framework::stride(collapsed_output_shape);

  // make a DFTI_DESCRIPTOR
  DftiDescriptor desc =
      _plan_mkl_fft(x->type(), out->type(), input_stride, output_stride,
                    signal_sizes, normalization, forward);
  if (forward) {
    MKL_DFTI_CHECK(DftiComputeForward(desc.get(), collapsed_input.data<void>(),
                                      collapsed_output.data<void>()));
  } else {
    MKL_DFTI_CHECK(DftiComputeBackward(desc.get(), collapsed_input.data<void>(),
                                       collapsed_output.data<void>()));
  }

  // resize for the collapsed output
  framework::DDim transposed_output_shape = out_sizes.transpose(dim_permute);
  collapsed_output.Resize(transposed_output_shape);
  framework::Tensor& transposed_output = collapsed_output;

  // reverse the transposition
  std::vector<int> reverse_dim_permute(ndim);
  for (int i = 0; i < ndim; i++) {
    reverse_dim_permute[dim_permute[i]] = i;
  }
  TransCompute<platform::CPUDeviceContext, To>(ndim, ctx, transposed_output,
                                               out, reverse_dim_permute);
}

template <typename Ti, typename To>
struct FFTC2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    exec_fft<platform::CPUDeviceContext, Ti, To>(ctx, x, out, axes,
                                                 normalization, forward);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward, bool onesided) {}
};

template <typename Ti, typename To>
struct FFTC2RFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {}
};

#elif defined(PADDLE_WITH_POCKETFFT)
template <typename Ti, typename To>
struct FFTC2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = typename Ti::value_type;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<int64_t> in_strides =
        framework::vectorize<int64_t>(framework::stride(input_dim));
    const int64_t data_size = sizeof(C);
    std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                   [](int64_t s) { return s * data_size; });

    const auto* in_data = reinterpret_cast<const C*>(x->data<Ti>());
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // well, we have to use std::vector<size_t> here
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet factor
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::c2c(in_sizes, in_strides, in_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

template <typename Ti, typename To>
struct FFTR2CFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward, bool onesided) {
    using R = Ti;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<int64_t> in_strides =
        framework::vectorize<int64_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<int64_t> out_strides =
        framework::vectorize<int64_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto* in_data = x->data<R>();
    auto* out_data = reinterpret_cast<C*>(out->data<To>());
    // well, we have to use std::vector<size_t> here
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet facet
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= in_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::r2c(in_sizes, in_strides, out_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

template <typename Ti, typename To>
struct FFTC2RFunctor<platform::CPUDeviceContext, Ti, To> {
  void operator()(const platform::CPUDeviceContext& ctx, const Tensor* x,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {
    using R = To;
    using C = std::complex<R>;

    const auto& input_dim = x->dims();
    const std::vector<size_t> in_sizes =
        framework::vectorize<size_t>(input_dim);
    std::vector<int64_t> in_strides =
        framework::vectorize<int64_t>(framework::stride(input_dim));
    {
      const int64_t data_size = sizeof(C);
      std::transform(in_strides.begin(), in_strides.end(), in_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto& output_dim = out->dims();
    const std::vector<size_t> out_sizes =
        framework::vectorize<size_t>(output_dim);
    std::vector<int64_t> out_strides =
        framework::vectorize<int64_t>(framework::stride(output_dim));
    {
      const int64_t data_size = sizeof(R);
      std::transform(out_strides.begin(), out_strides.end(),
                     out_strides.begin(),
                     [](int64_t s) { return s * data_size; });
    }

    const auto* in_data = reinterpret_cast<const C*>(x->data<To>());
    auto* out_data = out->data<R>();
    // well, we have to use std::vector<size_t> here
    std::vector<size_t> axes_(axes.size());
    std::copy(axes.begin(), axes.end(), axes_.begin());
    // compuet facet
    int64_t signal_numel = 1;
    for (auto i : axes) {
      signal_numel *= out_sizes[i];
    }
    R factor = compute_factor<R>(signal_numel, normalization);
    pocketfft::c2r(out_sizes, in_strides, out_strides, axes_, forward, in_data,
                   out_data, factor);
  }
};

#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fft_c2c, ops::FFTC2COp, ops::FFTC2COpMaker,
                  ops::FFTC2CGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTC2CGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2c_grad, ops::FFTC2CGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_r2c, ops::FFTR2COp, ops::FFTR2COpMaker,
                  ops::FFTR2CGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTR2CGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_r2c, ops::FFTR2CKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTR2CKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_r2c_grad, ops::FFTR2CGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_r2c_grad,
    ops::FFTR2CGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTR2CGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2r, ops::FFTC2ROp, ops::FFTC2ROpMaker,
                  ops::FFTC2RGradOpMaker<paddle::framework::OpDesc>,
                  ops::FFTC2RGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fft_c2r, ops::FFTC2RKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2RKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(fft_c2r_grad, ops::FFTC2RGradOp);
REGISTER_OP_CPU_KERNEL(
    fft_c2r_grad,
    ops::FFTC2RGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FFTC2RGradKernel<paddle::platform::CPUDeviceContext, double>);
