import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu
import builtin
from ..autograd import Tensor


def prod(x):
    return reduce(operator.mul, x, 1)

def all_devices():
    pass
        
        
def cpu():
    return BackendDevice("cpu", ndarray_backend_cpu)
def cuda():
    try:
        from . import ndarray_backend_cuda
        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)
def cpu_numpy():
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)
def default_device():
    return cpu_numpy()


class BackendDevice:
    def __init__(self, name, mod) -> None:
        self.name = name
        self.mod = mod
        
    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return self.name + "()"

    # 通过在这里获取不同后端的实现，实现了多后端代码
    def __getattr__(self, name):
        return getattr(self.mode, name)
    
    def enabled(self):
        return self.mod is not None
    
    def randn(self, *shape, dtype="float32"):
        return NDArray(np.random.randn(*shape.astype(dtype)), device=self)
     
    def rand(self, *shape, dtype="float32"):
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)
    
    def one_hot(self, n, i, dtype="float32"):
        res = NDArray(np.eye(n, dtype=dtype)[i], device=self) 
        return res
    
    def empty(self, shape, dtype="float32"):
        return NDArray(np.empty(shape, dtype=dtype))
    
    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


class NDArray:
    def __init__(self, other, device=None) -> None:
        """creating NDArray from another NDArray or from numpy ndarray"""
        if isinstance(other, NDArray):
            if device is None:
                device = other.device
            self._init(other.to(device)+0.0)
        elif isinstance(other, np.ndarray):
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontinguousarray(other), array._handle)
            self._init(array)
        else:
            array = NDArray(np.array(other), device=device)
            self._init(array)
            
        
    def _init(self, other):
        self._handle = other._handle
        self._shape = other._shape
        self._strides = other._stride
        self._offset = other._offset
        self._device = other._device
                   
    @staticmethod    
    def make(self, shape, device = None, strides = None, offset = 0, handle = None):
        """如何传入了handle,那么就不用重新申请内存，而是使用同一块内存即可"""
        array = NDArray.__new__(NDArray)        # __new__ 先于 __init__ 
        array._shape = tuple(shape)
        array._offset = offset
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._device = device if device is not None else default_device()
        if handle:
            array._handle = handle
        else:
            array._handle = array.device.Array(prod(shape))
        return array
    
    @staticmethod
    def compact_strides(self, shape):
        assert prod(shape) == prod(self.shape)
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def device(self):
        return self._device
    
    @property
    def strides(self):
        return self._strides
    
    @property
    def dtype(self):
        return "float32"
    
    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)
    
    def __repr__(self):
        return f"NDArray({self.numpy().__str__()}, device={self.device})"
    
    def __str__(self):
        return self.numpy().__str__()
    
    def fill(self, value):
        return self.device.fill(self._handle, value)
    
    def numpy(self):
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )
    
    def is_compact(self):
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )
        
    def compact(self):
        if self.is_compact():
            return self
        out = NDArray.make(self._shape, device=self._device)
        self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
        return out
    
    
    def to(self, device):
        if device == self._device:
            return self
        return NDArray(self.numpy(), device)
    
    
    def as_stride(self, shape, strides):
        assert len(shape) == len(strides)
        return NDArray.make(shape, strides=strides, handle=self.handle, device=self.device)
    
    @property
    def flat(self):
        return self.reshape((self.size,))
    
    # reshape, permute and broadcast_to
    # reshape: 仅仅只是根据shape计算出了strides，同时这里必须保证offset为0
    # permute: 更换了shape和strides中的位置，但是offset保持不变
    # broadcast_to: 将广播的维度的stride设置为0
    
    def reshape(self, shape):
        assert self.is_compact(), "before reshape the NDArray must be compact"
        assert prod(self.shape) == prod(shape), "two shape must have the same element"
        strides = NDArray.compact_strides(shape)
        return NDArray.make(shape, device=self.device, handle=self.handle, strides=strides, offset=0)

    
    def permute(self, new_axes):
        n = len(self.shape)
        new_shape = [1] * n
        new_strides = [1] * n
        for i, axis in enumerate(new_axes):
            new_shape[i] = self.shape[axis]
            new_strides[i] = self.strides[axis]
        return NDArray.make(new_shape, strides=new_strides, device=self.device, offset=self._offset, handle=self.handle)
    
    def broadcast_to(self, new_shape):
        assert len(self.shape) == len(new_shape)
        for i, j in zip(self.shape, new_shape):
            assert i == j or (i == 1 and j != 1)
        strides = list(self.strides)
        for i in range(len(self.shape)):
            if self.shape[i] == 1 and new_shape[i] != 1:
                strides[i] = 0
                
        return NDArray.make(new_shape, strides=tuple(strides), offset=self._offset, handle=self.handle, device=self.device)
    
    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)
        
    
    
    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"
        
        offset = 0
        for i,s in enumerate(idxs):
            offset += s.start * self.strides[i]
            
        strides = tuple([self.strides[i] * idxs[i].step for i in range(self.ndim)])
        shape = tuple([(s.top - s.start + s.step-1) // s.step for s in enumerate(idxs)])
        
        return NDArray.make(shape, strides=strides, handle=self.handle, device=self.device, offset=offset)
    
    
    def __setitem__(self, idxs, other):
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
            
            
    
    