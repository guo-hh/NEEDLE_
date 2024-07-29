"""先使用numpy作为后端实现自动微分等功能
然后再使用自定义后端
"""

import numpy

class Device:
    """Baseclass data that sits in CPU"""

class CPUDevice(Device):
    def __repr__(self):
        return "neeld.cpu()"
    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True
    
    def zeros(self, *shape, dtype='float32'):
        return numpy.zeros(shape, dtype=dtype)
    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)
    
    # 这里一定要加*
    def randn(self, *shape):
        return numpy.random.randn(*shape)
    def rand(self, *shape):
        return numpy.random.rand(*shape)
    
    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]
    
    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)
    
    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)
    
def cpu():
    return CPUDevice()
def all_devices():
    return [cpu()]
def default_device():
    return cpu()