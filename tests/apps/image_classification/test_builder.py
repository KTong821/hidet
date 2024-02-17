import pytest
import torch
from datasets import load_dataset
from hidet.apps.image_classification.builder import create_image_classifier
from hidet.graph.tensor import from_torch
from transformers import AutoImageProcessor


def test_create_image_classifier():
    dataset = load_dataset("imagenet-1k", split="validation[0:1]", trust_remote_code=True)

    # using huggingface pre-processor
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    image = image_processor(dataset[0]["image"], return_tensors="pt")["pixel_values"]
    image = from_torch(image).cuda()

    resnet = create_image_classifier("microsoft/resnet-50")
    assert "resnet" in resnet.compiled_app.meta.graphs
    assert resnet.compiled_app.meta.name == "microsoft/resnet-50"

    res = resnet.compiled_app.graphs["resnet"].run_async([image])
    res = res[0].torch()
    res = torch.argmax(res, dim=1)
    
    assert res[0] == dataset[0]["label"]


if __name__ == "__main__":
    pytest.main([__file__])
