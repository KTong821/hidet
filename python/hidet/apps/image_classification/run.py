from typing import Dict, List
from transformers import ResNetForImageClassification, AutoImageProcessor
from hidet.graph.tensor import Tensor, from_torch
import torch
import numpy as np
from datasets import load_dataset, config, Dataset
from hidet.apps.image_classification.builder import create_image_classifier
from tqdm import tqdm

import matplotlib.pyplot as plt

config.IN_MEMORY_MAX_SIZE = int(1e11)
bs = 50


def transform_tensor(batch: Dict[str, List], *extra_args) -> Dict[str, List]:
    batch["input_tensor"] = []
    for image in batch["image"]:
        t = torch.tensor(np.asarray(image.resize((224, 224))))

        if len(t.shape) == 2:
            # grayscale, cast to RGB
            t = t.expand(3, -1, -1).contiguous()
        else:
            # channel last RGB, permute to channel first
            t = t.permute(2, 0, 1).contiguous()
        batch["input_tensor"].append(t)
    
    return batch


def main():
    dataset: Dataset = load_dataset(
        "imagenet-1k", split="validation", trust_remote_code=True
    )
    dataset = dataset.with_transform(transform_tensor, output_all_columns=True)

    batches: list[Tensor] = []
    for pos in tqdm(range(0, len(dataset), bs)):
        batch = torch.stack(dataset[pos:pos + bs]["input_tensor"])
        print(batch.shape)
        batch = from_torch(batch).to(dtype="float32", device="cuda")
        batches.append(batch)
        break

    # image_classifier = create_image_classifier("microsoft/resnet-50")

    # print(len(batches))
    # print(batches[0].shape)
    # res = image_classifier.classify(batches)
    # res = res[0].torch()
    # print(res)
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    inputs = image_processor(dataset[0:50]["image"], return_tensors="pt")
    print(inputs.keys())
    print(type(inputs))
    print(inputs["pixel_values"].shape)
    print(inputs["pixel_values"])

    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").cuda()

    with torch.no_grad():
        outputs = model(inputs["pixel_values"])#.logits
        print(type(outputs))
        print(outputs)


    predicted_label = torch.argmax(outputs, dim=1)
    print(predicted_label)
    # print(model.config.id2label[predicted_label])
    # print(dataset)
    # print(dataset[0])
    # print(dataset[0]["input_tensor"].shape)
    # print(torch.stack(0))
    # print(dataset[:50]["input_tensor"])

    # plt.imshow(dataset[0]['input_image'].permute(1, 2, 0))
    # plt.show()


if __name__ == "__main__":
    main()
