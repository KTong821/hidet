from typing import Optional
import hidet
from transformers import PretrainedConfig, AutoConfig

def _load_pretrained_config(model: str, revision: Optional[str] = None) -> PretrainedConfig:
    try:
        huggingface_token = hidet.option.get_option('tokens.for_huggingface')
        return AutoConfig.from_pretrained(model, revision=revision, token=huggingface_token)
    except ValueError as e:
        raise e