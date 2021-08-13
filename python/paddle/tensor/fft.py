# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from .attribute import is_complex, is_floating_point
from .framework import in_dygraph_mode
from .. import _C_ops


# public APIs 1d
def fft(x, n=None, axis=-1, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, n, axis, norm, forward=True, onesided=False)
    else:
        return fft_c2c(x, n, axis, norm, forward=True)


def ifft(x, n=None, axis=-1, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, n, axis, norm, forward=False, onesided=False)
    else:
        return fft_c2c(x, n, axis, norm, forward=False)


def rfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_r2c(x, n, axis, norm, forward=True, onesided=True)


def irfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_c2r(x, n, axis, norm, forward=False)


def hfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_c2r(x, n, axis, norm, forward=True)


def ihfft(x, n=None, axis=-1, norm=None, name=None):
    return fft_r2c(x, n, axis, norm, forward=False, onesided=True)


# public APIs nd
def fftn(x, s=None, axes=None, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, s, axes, norm, forward=True, onesided=False)
    else:
        return fft_c2c(x, s, axes, norm, forward=True)


def ifftn(x, s=None, axes=None, norm=None, name=None):
    if is_floating_point(x):
        return fft_r2c(x, s, axes, norm, forward=False, onesided=False)
    else:
        return fft_c2c(x, s, axes, norm, forward=False)


def rfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_r2c(x, s, axes, norm, forward=True, onesided=True)


def irfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_c2r(x, s, axes, norm, forward=False)


def hfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_c2r(x, s, axes, norm, forward=True)


def ihfftn(x, s=None, axes=None, norm=None, name=None):
    return fft_r2c(x, s, axes, norm, forward=False, onesided=True)


## public APIs 2d
def fft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return fftn(x, s, axes, norm, name)


def ifft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return ifftn(x, s, axes, norm, name)


def rfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return rfftn(x, s, axes, norm, name)


def irfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return irfftn(x, s, axes, norm, name)


def hfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return hfftn(x, s, axes, norm, name)


def ihfft2(x, s=None, axes=(-2, -1), norm=None, name=None):
    return ihfftn(x, s, axes, norm, name)


# public APIs utilities
def fftfreq(n, d=1.0, dtype=None, name=None):
    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = (n + 1) // 2
    neg_max = n // 2
    indices = paddle.arange(-neg_max, pos_max, dtype=dtype, name=name)
    indices = paddle.roll(indices, -neg_max, name=name)
    return indices * val


def rfftfreq(n, d=1.0, dtype=None, name=None):
    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = 1 + n // 2
    indices = paddle.arange(0, pos_max, dtype=dtype, name=name)
    return indices * val


def fftshift(x, axes=None, name=None):
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = paddle.rank(x).reshape([1])
        axes = axes or paddle.arange(0, rank)
        shifts = [size // 2 for size in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = [shape[ax] // 2 for ax in axes]
    return paddle.roll(x, shifts, axes, name=name)


def ifftshift(x, axes=None, name=None):
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = paddle.rank(x).reshape([1])
        axes = axes or paddle.arange(0, rank)
        shifts = [-size // 2 for size in shape]
    elif isinstance(axes, int):
        shifts = -shape[axes] // 2
    else:
        shifts = [-shape[ax] // 2 for ax in axes]
    return paddle.roll(x, shifts, axes, name=name)





# internal functions
def fft_c2c(x, n, axis, norm, forward):
    # TODO, move error checking to operators
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError("Unexpected norm: {}. Norm should be forward, backward or ortho".form(norm))
    rank = paddle.rank(x)
    if axis < -rank or axis >= rank:
        raise ValueError("Invalid axis. Input's ndim is {}, axis should be [-{}, {})".format(rank, rank, rank))
    if axis < 0:
        axis += rank
    s = [axis]
    axes = [axis]
    op_type = 'fft_c2c'

    if in_dygraph_mode():
        attrs = ('s', s, 'axes', axes, 'norm', norm, 'forward', forward)
        out= getattr(_C_ops, "fftc2c")(x, *attrs)
    else:
        op_type 
        inputs = {'X': [x], }
        attrs = {
            's': s,
            'axes': axes,
            'norm': norm,
            'forward': forward
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out
    
    


def fft_r2c(x, n, axis, norm, forward, onesided):
    pass


def fft_c2r(x, n, axis, norm, forward):
    pass


def fftn_c2c(x, s, axes, norm, forward):
    # TODO, move error checking to operators
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError("Unexpected norm: {}. Norm should be forward, backward or ortho".form(norm))
    rank = paddle.rank(x)
    if axes is None
        if s is None:
            axes = list(range(rank))
            s = paddle.shape(x)
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        axes_ = axes.copy()
        for i in len(axes_):
            if axes_[i] < -rank or axes_[i] >= rank:
                raise ValueError("Invalid axis. Input's ndim is {}, axis should be [-{}, {})".format(rank, rank, rank))
            if axes_[i] < 0:
                axes_[i] += rank
        axes = axes_
        if s is None:
            shape = paddle.shape(x)
            s = [shape[axis] for axis in axes]
        else:
            assert len(axes) == len(s)
    op_type = 'fft_c2c'

    if in_dygraph_mode():
        attrs = ('s', s, 'axes', axes, 'norm', norm, 'forward', forward)
        out= getattr(_C_ops, "fftc2c")(x, *attrs)
    else:
        op_type 
        inputs = {'X': [x], }
        attrs = {
            's': s,
            'axes': axes,
            'norm': norm,
            'forward': forward
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out

            
    


    
    s = [axis]
    axes = [axis]
    op_type = 'fft_c2c'

    if in_dygraph_mode():
        attrs = ('s', s, 'axes', axes, 'norm', norm, 'forward', forward)
        out= getattr(_C_ops, "fftc2c")(x, *attrs)
    else:
        op_type 
        inputs = {'Input': [x], }
        attrs = {
            's': s,
            'axes': axes,
            'norm': norm,
            'forward': forward
        }
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 op_type)
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)


def fftn_r2c(x, s, axes, norm, forward, onesided):
    pass


def fftn_c2r(x, s, axes, norm, forward):
    pass
