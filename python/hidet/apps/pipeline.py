

from typing import Any, Iterable, Optional, Sequence

from hidet.apps.processing import BaseProcessor
from hidet.graph.tensor import Tensor
from tqdm import tqdm


class Pipeline:

    def __init__(self, name: str, revision: Optional[str], pre_processor: BaseProcessor):
        self.name = name
        self.revision = revision
        self.pre_processor = pre_processor

    def preprocess(self, model_inputs: Any, **kwargs):
        raise NotImplementedError("Pipeline subclasses should implement their own preprocess step.")
    
    def postprocess(self, model_outputs: Tensor, **kwargs):
        raise NotImplementedError("Pipeline subclasses should implement their own postprocess step.")

    def forward(self, model_inputs: Tensor, **kwargs):
        raise NotImplementedError("Pipeline subclasses should implement their own forward step.")
    
    def __call__(self, model_inputs: Any, batch_size: int = 1, **kwargs):
        # TODO take iterable model_input

        if not isinstance(model_inputs, Iterable):
            model_inputs = [model_inputs]
        if not isinstance(model_inputs, Sequence):
            model_inputs = list(model_inputs)

        assert isinstance(model_inputs, Sequence)

        processed_inputs = self.preprocess(model_inputs, **kwargs)

        model_outputs = self.forward(processed_inputs, **kwargs)
        outputs = self.postprocess(model_outputs, **kwargs)

        return outputs

        # for i in tqdm(range(0, len(model_inputs), batch_size)):
        #     batch = processed_inputs[i:i+batch_size]
        #     model_outputs = self.forward(batch, **kwargs)
        #     outputs = self.postprocess(model_outputs, **kwargs)
        #     yield outputs
            # yield self.run(model_inputs[i:i+batch_size], **kwargs)
        

    # def run(self, model_inputs: Any, **kwargs):
    #     model_inputs = self.preprocess(model_inputs, **kwargs)
    #     model_outputs = self.forward(model_inputs, **kwargs)
    #     outputs = self.postprocess(model_outputs, **kwargs)
    #     return outputs
