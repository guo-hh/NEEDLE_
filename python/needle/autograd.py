
import needle
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy
from .backend_selection import NDArray, array_api, cpu
from needle import init

LAZY_MODE = False
TENSOR_COUNTER = 0



class Op:
    def __call__(self, *args):
        raise NotImplementedError()
    
    def compute(self, *args: Tuple[NDArray]):
        raise NotImplementedError()
    
    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint"""
        raise NotImplementedError()
        
    def gradient_as_tuple(self, out_grad:"Value", node: "Value") -> Tuple["Value"]:
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        return (output, )
    
    
class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    
class TensorTupleOp(Op):
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)

class Value:
    """A value in the computational graph"""
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool
    
    def realize_cached_data(self):
        # 真正要使用的时候，才进行初始化
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def is_leaf(self):
        return self.op is None
    
    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1
    
    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *, 
        num_outputs: int=1,
        cached_data: List[object] = None,
        requires_grad: bool = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data  = cached_data
        self.requires_grad = requires_grad
        
    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value
    
    
    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)
        
        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()       ## detach的好处是什么？？
            value.realize_cached_data()
            
        return value
    


# TensorTuple为什么要定义这样一个类呢?
# TensorTuple的cached_data为tuple, 关于为什么要定义这样一个类，我的猜想是因为stack的操作对象为一个tuple
class TensorTuple(Value):
    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)
    def __getitem__(self, index:int):
        return needle.ops.tuple_get_item(self, index)
    def tuple(self):
        return tuple([x for x in self])
    
    def __repr__(self):
        return "needle.TensorTuple " + str(self.tuple())
    
    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())
    
    
    

class Tensor(Value):
    grad: "Tensor"
    
    def __init__(self,
        array,  
        *,
        device=None,
        dtype=None,
        requires_grad=True,
        **kwargs
        ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(array.numpy(), device=device, dtype=dtype)
        else:
            device = device if device is not None else cpu()
            # print(f"Tensor init: device: {device}, dtype: {dtype}")
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            # print("array_api is numpy")
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)
    
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()      ## 有什么作用
            tensor.realize_cached_data()
        return tensor
    
    @staticmethod
    def make_const(data, requires_grad=False):
        """
        data is not Tensor but is a cached_data
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data = data if not isinstance(data, Tensor) else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()
    
    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert isinstance(value.dtype==self.dtype, f"{value.dtype} {self.dtype}")
        self.cached_data = value.realize_cached_data()
    
    
    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph"""
        res = Tensor.make_const(self.realize_cached_data())
        return res
    
    @property
    def shape(self):
        return self.realize_cached_data().shape
    @property
    def dtype(self):
        return self.realize_cached_data().dtype
    @property
    def device(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad=(
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )       # 这里为什么要写成tuple的形式
        compute_gradiend_of_variables(self, out_grad)
        
    def __repr__(self):
        return "needle.Tensor("+str(self.realize_cached_data())+")"
    
    def __str__(self):
        return self.realize_cached_data().__str__()
    
    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()
    
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)
        
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)
        
    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)
        
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)
    
    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)
    
    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)
    
    def matmul(self, other):
        return needle.ops.MatMul()(self, other)
    
    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)
    
    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)
    
    def __neg__(self):
        return needle.ops.Negate()(self)
    
    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)
    
    
    # 该成员函数只有在使用nd作为后端时才有用
    def to(self, device):
        if self.device == device:
            return self
        self.cached_data = self.cached_data.to(device)
        return self
    
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__
    
        

    
    
def compute_gradiend_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[output_tensor] = [out_grad]
    
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    
    for node in reverse_topo_order:
        adjoint = sum_node_list(node_to_output_grads_list[node])        # 多路径梯度相加
        node.grad = adjoint
        if node.op is None:
            continue
        partial_grads = node.op.gradient_as_tuple(adjoint, node)
        for input, partial_grad in zip(node.inputs, partial_grads):
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = []
            node_to_output_grads_list[input].append(partial_grad)
            

def find_topo_sort(node_list: List["Value"]) -> List["Value"]:
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for input in node.inputs:
        topo_sort_dfs(input, visited, topo_order)
    topo_order.append(node)
    

def sum_node_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
        
    
        
            
        
    
    