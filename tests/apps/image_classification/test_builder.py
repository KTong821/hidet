from hidet.apps.image_classification.processing.image_processor import ChannelDimension
import pytest
import torch
from datasets import load_dataset
from hidet.apps.image_classification.builder import create_image_classifier, create_image_processor
from hidet.graph.tensor import from_torch
from transformers import AutoImageProcessor


def test_create_image_classifier():
    # load 64 images
    dataset = load_dataset("imagenet-1k", split="validation[0:64]", trust_remote_code=True)

    # using huggingface pre-processor
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    images = image_processor(dataset["image"], return_tensors="pt")
    images = images["pixel_values"]
    images = from_torch(images).cuda()

    resnet = create_image_classifier("microsoft/resnet-50", batch_size=64, kernel_search_space=0)
    assert "resnet" in resnet.compiled_app.meta.graphs
    assert resnet.compiled_app.meta.name == "microsoft/resnet-50"

    res = resnet.compiled_app.graphs["resnet"].run_async([images])
    res = res[0].torch()
    res = torch.argmax(res, dim=1)

    labels = torch.tensor(dataset["label"][:64]).cuda()
    accuracy = torch.sum(res == labels) / 64
    assert accuracy > 0.8


def test_create_image_processor():
    # load 64 images
    dataset = load_dataset("imagenet-1k", split="validation[0:64]", trust_remote_code=True)

    # use our preprocessor
    image_processor = create_image_processor("microsoft/resnet-50")

    images = image_processor(dataset["image"], input_data_format=ChannelDimension.CHANNEL_LAST)

    assert images.shape == (64, 3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__])
