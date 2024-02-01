from typing import List, Optional
import hidet
from hidet.apps.image_classification.app import ResNet
from hidet.apps.image_classification.modeling.pretrained import (
    PretrainedModelForImageClassification,
)
from hidet.apps.modeling_outputs import ImageClassifierOutput
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.tensor import Tensor, symbol
from hidet.runtime.compiled_app import create_compiled_app
from transformers import PretrainedConfig

from hidet.apps import hf
from hidet.graph import trace_from


def create_image_classifier(
    name: str,
    revision: Optional[str] = None,
    dtype: Optional[str] = None,
    device: str = "cuda",
    kernel_search_space: int = 0,
):
    # load the huggingface config according to (model, revision) pair
    config: PretrainedConfig = hf.load_pretrained_config(name, revision=revision)

    # load model instance by architecture, assume only 1 architecture for now
    model = PretrainedModelForImageClassification.create_pretrained_model(
        config, revision=revision, dtype=dtype, device=device
    )
    inputs: Tensor = symbol(["bs", 3, "h", "w"], dtype="float32", device=device)
    outputs: ImageClassifierOutput = model.forward(inputs)
    graph: FlowGraph = trace_from(outputs.logits, inputs)

    with open("./resnet_graph.json", "w") as f:
        hidet.utils.netron.dump(graph, f)

    compiled_graph = graph.build(space=kernel_search_space)

    return ResNet(
        compiled_app=create_compiled_app(
            graphs={"resnet": compiled_graph}, name="ResNet"
        )
    )
