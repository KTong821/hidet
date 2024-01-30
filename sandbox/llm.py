# from transformers import AutoConfig

# config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# print(config)


# from transformers.modeling_outputs import BaseModelOutput

# a = BaseModelOutput()

# print(str(a))


from transformers import AutoImageProcessor

import torch

from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

print(image_processor)

i = image_processor(image, return_tensors="pt")

print(i["pixel_values"].shape)

from transformers import ConvNextImageProcessor