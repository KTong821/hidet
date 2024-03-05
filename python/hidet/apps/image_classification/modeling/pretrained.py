from typing import Optional

import torch
from hidet.apps.modeling_outputs import ImageClassifierOutput
from hidet.apps.pretrained import PretrainedModel
from transformers import AutoModelForImageClassification, PretrainedConfig
from transformers import PreTrainedModel as TransformersPretrainedModel

import hidet


class PretrainedModelForImageClassification(PretrainedModel[ImageClassifierOutput]):
    @classmethod
    def create_pretrained_model(
        cls, config: PretrainedConfig, revision: Optional[str] = None, dtype: Optional[str] = None, device: str = "cuda"
    ):
        # dynamically load model subclass
        architectures = getattr(config, "architectures")
        if not architectures:
            raise ValueError(f"Config {config.name_or_path} has no architecture.")
        pretrained_model_class = cls.load_module(architectures[0])

        # load the pretrained huggingface model
        huggingface_token = hidet.option.get_option("auth_tokens.for_huggingface")
        with torch.device("cuda"):  # reduce the time to load the model
            torch_model: TransformersPretrainedModel = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path=config.name_or_path,
                torch_dtype=torch.float32,
                revision=revision,
                token=huggingface_token,
            )

        torch_model = torch_model.cpu()
        torch.cuda.empty_cache()

        dtype = cls.parse_dtype(config)
        hidet_model = pretrained_model_class(config)
        hidet_model.to(dtype=dtype, device=device)

        cls.copy_weights(torch_model, hidet_model)

        return hidet_model
