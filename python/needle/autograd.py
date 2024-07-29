
import needle
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy
from .backend_selection import NDArray, array_api


LAZY_MODE = False
TENSOR_COUNTER = 0



class Tensor:
    def __init__(self) -> None:
        pass