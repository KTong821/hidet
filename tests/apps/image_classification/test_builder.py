from hidet.apps.image_classification.processing.image_processor import ChannelDimension
import pytest
import torch
from datasets import load_dataset
from hidet.apps.image_classification.builder import create_image_classifier, create_image_processor
from hidet.graph.tensor import from_torch
from transformers import AutoImageProcessor


def test_create_image_classifier():
    dataset = load_dataset("huggingface/cats-image", split="test", trust_remote_code=True)

    # using huggingface pre-processor
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    images = image_processor(dataset["image"], return_tensors="pt")
    images = images["pixel_values"]
    images = from_torch(images).cuda()

    resnet = create_image_classifier("microsoft/resnet-50", kernel_search_space=0)
    assert "image_classifier" in resnet.compiled_app.meta.graphs
    assert resnet.compiled_app.meta.name == "microsoft/resnet-50"

    res = resnet.compiled_app.graphs["image_classifier"].run_async([images])
    res = res[0].torch()
    res = torch.argmax(res, dim=1)

    assert res[0].item() == 282  # tiger cat label


if __name__ == "__main__":
    pytest.main([__file__])
