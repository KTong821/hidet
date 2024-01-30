from typing import Optional
import hidet
from transformers import PretrainedConfig, AutoConfig


def load_pretrained_config(model: str, revision: Optional[str] = None) -> PretrainedConfig:
    huggingface_token = hidet.option.get_option('auth_tokens.for_huggingface')
    return AutoConfig.from_pretrained(model, revision=revision, token=huggingface_token)
