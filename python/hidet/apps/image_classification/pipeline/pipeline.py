from typing import Any, Optional
from hidet.apps import hf
from hidet.apps.image_classification.builder import create_image_classifier, create_image_processor
from hidet.apps.image_classification.processing.image_processor import BaseImageProcessor, ChannelDimension, ImageInput
from hidet.apps.pipeline import Pipeline
from hidet.graph.ops import softmax
from hidet.graph.ops.reduce.reduce import argmax
from hidet.graph.tensor import Tensor


class ImageClassificationPipeline(Pipeline):

    def __init__(self, name: str, revision: Optional[str] = None, pre_processor: Optional[BaseImageProcessor] = None, dtype: str = "float32", device: str = "cuda"):
        if pre_processor is None:
            image_processor = create_image_processor(name, revision)
        else:
            image_processor = pre_processor
        super().__init__(name, revision, image_processor)

        self.model = create_image_classifier(name, revision, dtype, device)
        self.config = hf.load_pretrained_config(name, revision)
        
    def __call__(
        self, images: ImageInput, **kwargs
    ):
        """
        Run through image classification pipeline end to end.

        images: ImageInput
            List or single instance of numpy array, PIL image, or torch tensor

        input_data_format: ChannelDimension
            Input data is channel first or last

        batch_size: int (default 1)
            Batch size to feed model inputs

        top_k: int (default 5)
            Return scores for top k results
        """
        return super().__call__(images, **kwargs)
    
    def preprocess(self, images: ImageInput, input_data_format: ChannelDimension, **kwargs):
        # TODO accept inputs other than ImageInput type, e.g. url or dataset

        return self.pre_processor(images, input_data_format=input_data_format, **kwargs)
    
    def postprocess(self, model_outputs: Tensor, top_k: int = 5, **kwargs):
        top_k = min(top_k, self.config.num_labels)
        torch_outputs = model_outputs.torch()
        values, indices = torch_outputs.topk(top_k, sorted=False)
        labels = [[self.config.id2label[int(x.item())] for x in t] for t in indices]
        return [[{"label": label, "score": value.item()} for label, value in zip(a, b)] for a, b in zip(labels, values)]
    
    def forward(self, model_inputs: Tensor, **kwargs) -> Tensor:
        return self.model.classify([model_inputs])[0]

