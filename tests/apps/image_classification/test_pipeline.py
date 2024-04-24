from hidet.apps.image_classification.pipeline.pipeline import ImageClassificationPipeline
from hidet.apps.image_classification.processing.image_processor import ChannelDimension
import pytest
from datasets import load_dataset


def test_image_classifier_pipeline():
    dataset = load_dataset("huggingface/cats-image", split="test", trust_remote_code=True)

    pipeline = ImageClassificationPipeline("microsoft/resnet-50", batch_size=1, kernel_search_space=0)

    res = pipeline(dataset["image"], input_data_format=ChannelDimension.CHANNEL_LAST, top_k=3)

    assert len(res) == 1
    assert all([len(x) == 3 for x in res])


if __name__ == "__main__":
    pytest.main([__file__])
