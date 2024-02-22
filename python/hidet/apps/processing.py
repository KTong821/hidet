from typing import Any

from hidet.apps.registry import Registry
from hidet.graph.tensor import Tensor


class BaseProcessor(Registry):
    def __call__(self, data: Any, **kwargs) -> Tensor:
        return self.preprocess(data, **kwargs)

    def preprocess(self, data: Any, **kwargs) -> Tensor:
        raise NotImplementedError("Processors should implement their own preprocess step.")
