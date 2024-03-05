from hidet.graph.tensor import from_torch

from typing import Optional
from hidet.apps.modeling_outputs import DiffusionOutput
from hidet.apps.pretrained import PretrainedModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import hidet
import torch

class PretrainedModelForText2Image(PretrainedModel[DiffusionOutput]):
    @classmethod
    def create_pretrained_model(
        cls,
        name: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "cuda",
    ):
        # dynamically load model subclass

        # load the pretrained huggingface model
        # note: diffusers pipeline is more similar to model
        huggingface_token = hidet.option.get_option("auth_tokens.for_huggingface")
        with torch.device(device):
            stable_diffusion_pipeline: DiffusionPipeline = (
                StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path=name,
                    torch_dtype=torch.float32,
                    revision=revision,
                    token=huggingface_token,
                )
            )

        pipeline_config = stable_diffusion_pipeline.config
        print(pipeline_config)

        torch_unet = stable_diffusion_pipeline.unet
        print("torch group norm")
        print(torch_unet.down_blocks[0].resnets[0].norm1.weight.shape)
        pretrained_unet_class = cls.load_module(pipeline_config["unet"][1])
        
        hidet_unet = pretrained_unet_class(**dict(torch_unet.config))
        hidet_unet.to(dtype=dtype, device=device)

        cls.copy_weights(torch_unet, hidet_unet)

        tensors = torch.load("unet_embeddings.pt")
        print(tensors)

        sample = tensors["sample"]
        timesteps = tensors["timesteps"]
        hidet_unet(from_torch(sample), from_torch(timesteps))


        return None

