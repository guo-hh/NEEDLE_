from . import ops
from .ops import *
from .autograd import Tensor  ## 还没有定义

from . import init
from .init import ones, zeros, zeros_like, ones_like
from .init import rand,randn
from . import data
from . import nn
from . import optim
from .backend_selection import *