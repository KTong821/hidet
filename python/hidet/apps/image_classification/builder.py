from typing import List, Optional
from hidet.apps.image_classification.modeling.pretrained import (
    PretrainedModelForImageClassification,
)
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.tensor import Tensor, symbol
from transformers import PretrainedConfig

from hidet.apps import hf
from hidet.ir.dtypes import float32
from hidet.graph import trace_from


def create_image_classifier(
    name: str,
    revision: Optional[str] = None,
    dtype: Optional[str] = None,
    default_memory_capacity: Optional[int] = None,
    device: str = "cuda",
):
    # load the huggingface config according to (model, revision) pair
    config: PretrainedConfig = hf.load_pretrained_config(name, revision=revision)

    # load model instance by architecture, assume only 1 architecture for now
    model = PretrainedModelForImageClassification.create_pretrained_model(
        config, revision=revision, dtype=dtype, device=device
    )

    inputs: Tensor = symbol(["bs", "c", "h", "w"], dtype="float32", device=device)
    outputs: Tensor = model.forward(inputs)
    graph: FlowGraph = trace_from(outputs, inputs)

    print(graph)

    return 0


if __name__ == "__main__":
    image_classifier = create_image_classifier("microsoft/resnet-50")
