from numbers import Number
from typing import Optional,List, Tuple, Union

from ..autograd import TensorOp, Op, Tensor, Value
from ..autograd import TensorTuple, TensorTupleOp
from ..backend_selection import array_api, NDArray
import numpy


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b:NDArray):
        return a + b
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a,b):
    return EWiseAdd()(a,b)

class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, other: NDArray):
        return self.scalar + other
    def gradient(self, out_grad: Tensor, node:Tensor):
        return out_grad

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b:NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = b * out_grad
        grad_b = a * out_grad
        return grad_a, grad_b
    
def multiply(a,b):
    return EWiseMul()(a,b)

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
        
    def compute(self, a:NDArray):
        return self.scalar* a
    def gradient(self, out_grad:Tensor, node: Tensor):
        return (self.scalar * out_grad,)

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    def compute(self, a:NDArray, b:NDArray):
        return a**b
    
    def gradient(self, out_grad:Tensor, node: Tensor):    
        a,b = node.inputs
        grad_a = b * (a ** (b-1)) * out_grad
        grad_b = log(a) * node * out_grad
        return grad_a, grad_b
    
def power(a, b):
    return EWisePow()(a,b)


class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
        
    def compute(self, a: NDArray):
        return a ** self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return (self.scalar * a ** (self.scalar-1)) * out_grad

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b:NDArray):
        return a / b
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs      # 都是Tensor
        grad_a = out_grad / b
        grad_b = -1 *( a / b ** 2) * out_grad
        return grad_a, grad_b
        
def divide(a,b):
    return EWiseDiv()(a,b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
        
    def compute(self, a: NDArray):
        return a / self.scalar
    
    def gradient(self, out_grad:Tensor, node: Tensor):
        return out_grad / self.scalar

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes
    
    def compute(self, a: NDArray):
        ndim = len(a.shape)
        if self.axes is None:
            ndim = len(a.shape)
            axes = (ndim-1, ndim-2)
        else:
            axes = self.axes
        return array_api.swapaxes(a, axes[0], axes[1])
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        
    def compute(self, a:NDArray):    
        if array_api is numpy:
            return array_api.reshape(a, self.shape)
        return a.compact().reshape(self.shape)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        origin_shape = a.shape
        return reshape(out_grad, origin_shape)

def reshape(a, shape):
    return Reshape(shape)(a)
    

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        
    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)
    
    def gradient(self, out_grad:Tensor, node:Tensor):
        sum_axis = None
        origin_shape = node.inputs[0].shape
        origin_shape = (1,) * (len(self.shape)-len(origin_shape)) + origin_shape
        assert len(origin_shape) == len(self.shape), "just support broadcast between two dim-equaled Tensor"
        if origin_shape == self.shape:
            return out_grad
        sum_axis = tuple([i for i in range(len(origin_shape)) if origin_shape[i] != self.shape[i]])
        
        return summation(out_grad,sum_axis).reshape(origin_shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)
        
        
class Summation(TensorOp):
    def __init__(self, axes:Optional[tuple]=None):
        self.axes = axes
    
    
    ## 在这里支持了多维度相加
    def compute(self, a: NDArray):
        if isinstance(self.axes, (list, tuple)):
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis)
        elif isinstance(self.axes, int) or self.axes is None:
            a = a.sum(self.axes)
        return a
    
    def gradient(self, out_grad:Tensor, node: Tensor):
        origin_shape = node.inputs[0].shape
        new_shape = list(origin_shape)
        if self.axes is None:
            for i in range(len(new_shape)):
                new_shape[i] = 1
        else:
            if isinstance(self.axes, tuple):
                for axis in self.axes:
                    new_shape[axis] = 1
            elif isinstance(self.axes, int):
                new_shape[self.axes] = 1
        return out_grad.reshape(new_shape).broadcast_to(origin_shape)
        

def summation(a, axes=None):
    return Summation(axes)(a)



class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a,b)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = out_grad @ b.transpose()
        grad_b = a.transpose() @ out_grad
        if len(grad_a.shape) > len(a.shape):
            diff = len(grad_a.shape) - len(a.shape)
            axes = tuple([i for i in range(diff)])
            grad_a = summation(grad_a, axes)
        if len(grad_b.shape) > len(b.shape):
            diff = len(grad_b.shape) - len(b.shape)
            axes = tuple([i for i in range(diff)])
            grad_b = summation(grad_b, axes)
        return grad_a, grad_b            
            
def matmul(a, b):
    return MatMul()(a,b)

class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a
    def gradient(self, out_grad, node):
        return -1 * out_grad
def negate(a):
    return Negate()(a)

class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)
    def gradient(self, out_grad, node):
        input = node.inputs[0]
        return out_grad / input
    
def log(a):
    return Log()(a)

class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)
    def gradient(self, out_grad, node):
        return out_grad * node
    
def exp(a):
    return Exp()(a)


class Relu():
    def compute(self, a):
        return array_api.maximum(a, 0)
    def gradient(self, out_grad, node):
        input = node.inputs[0]
        mask = Tensor(input > 0, dtype=node.dtype, device=node.device, requires_grad=True)
        
        return mask * out_grad
    
def relu(a):
    return Relu()(a)


class Tanh():
    def compute(self, a):
        return array_api.tanh(a)
    
    def gradient(self, out_grad, node):
        return out_grad * (1 - node ** 2)

def tanh(a):
    return Tanh()(a)

class Sigmod():
    def compute(self, a):
        return 1 / (1 + array_api.exp(-a))
    def gradient(self, out_grad, node):
        return (1 - node) * node * out_grad

def sigmod(a):
    return Sigmod()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        self.axis = axis
    
    def compute(self, args: TensorTuple) -> Tensor:
        # args并不是TensorTuple而是NDArray
        # 该函数接受tuple of NDArray 返回NDArray
        assert len(args) > 0, "Stack needs at least one array !"
        shape = args[0].shape
        for arg in args:
            assert shape == arg.shape, "All arrays need to be of the same size!"
        n = len(args)
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        if array_api is numpy:
            out = array_api.empty(new_shape)
        else:
            out = array_api.empty(new_shape, device=args[0].device, dtype=args[0].dtype)
        slices = [slice(0,s) for i,s in enumerate(new_shape)]
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i+1)
            out[tuple(slices)] = arr
        return out
    
    def gradient(self, out_grad, node):
        return split(self.axis, out_grad)
    
def stack(axis, a):
    return Stack(axis)(a)
        
class Split(TensorTupleOp):
    def __init__(self, axis: int):
        self.axis = axis
    
    def compute(self, a: NDArray):
        # 该函数应该接受NDArray 返回tuple of NDArray
        out = []
        shape = a.shape
        n = shape[self.axis]
        slices = [slice(0,s) for s in enumerate(shape)]
        shape.pop(self.axis)
        for i in range(n):
            slices[i] = slices(i, i+1)
            tmp = a[tuple(slices).compact().reshape(shape)]
            out.append(tmp)
        return tuple(out)
        
    def gradient(self, out_grad, node):
        return stack(self.axis, out_grad)
        
def split(axis, a):
    return Split(axis)(a)
                

    
    