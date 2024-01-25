from typing import Optional
from transformers import PretrainedConfig

from hidet.apps import hf

def create_image_classifier(
    name: str,
    revision: Optional[str] = None,
    dtype: Optional[str] = None,
    default_memory_capacity: Optional[int] = None,
    device: str = 'cuda',
):
    config: PretrainedConfig = hf._load_pretrained_config(name, revision=revision)

    print(config)

if __name__ == "__main__":
    image_classifier = create_image_classifier("microsoft/resnet-50")
