from hidet.apps.image_classification.processing.image_processor import ChannelDimension
from hidet.apps.image_classification.processing.resnet.processing import ResNetImageProcessor
import pytest
import torch


def test_resnet_processor_resize():
    # Channel first
    image = torch.zeros((3, 10, 15), dtype=torch.uint8)
    image += torch.arange(1, 16)

    processor = ResNetImageProcessor(size=4, do_rescale=False, do_normalize=False)
    res = processor(image, input_data_format=ChannelDimension.CHANNEL_FIRST)
    assert res.shape == (1, 3, 4, 4)
    assert ((0 < res.torch()) & (res.torch() < 15)).all()

    # Channel last
    image = torch.zeros((10, 15, 3), dtype=torch.uint8)
    image += torch.arange(1, 16).view(1, 15, 1)

    processor = ResNetImageProcessor(size=4, do_rescale=False, do_normalize=False)
    res = processor(image, input_data_format=ChannelDimension.CHANNEL_LAST)
    assert res.shape == (1, 3, 4, 4)
    assert ((0 < res.torch()) & (res.torch() < 15)).all()

    # Batch resize
    images = []

    import random

    for _ in range(10):
        rows = random.randint(10, 20)
        cols = random.randint(10, 20)
        tensor = torch.randint(1, 9, (rows, cols, 3), dtype=torch.uint8)
        images.append(tensor)

    res = processor(images, input_data_format=ChannelDimension.CHANNEL_LAST)
    assert res.shape == (10, 3, 4, 4)
    assert ((0 <= res.torch()) & (res.torch() < 10)).all()


if __name__ == "__main__":
    pytest.main([__file__])
