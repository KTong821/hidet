from typing import Dict
from hidet import nn
from hidet.apps.diffusion.modeling.stable_diffusion.timestep import (
    TimestepEmbedding,
    Timesteps,
)
from hidet.apps.diffusion.modeling.stable_diffusion.unet_blocks import (
    CrossAttnDownBlock2D,
)
from hidet.apps.modeling_outputs import UNet2DConditionOutput
from hidet.apps.pretrained import PretrainedModel
from hidet.apps.registry import ModuleType, RegistryEntry
from hidet.graph.tensor import Tensor
from hidet.graph.ops import broadcast

import hidet

PretrainedModel.register(
    module_type=ModuleType.MODEL,
    arch="UNet2DConditionModel",
    entry=RegistryEntry(
        model_category="diffusion",
        module_name="stable_diffusion",
        klass="UNet2DConditionModel",
    ),
)


class UNet2DConditionModel(PretrainedModel):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        print("INIT UNET")
        self.conv_in = nn.Conv2d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["block_out_channels"][0],
            kernel_size=kwargs["conv_in_kernel"],
            padding=(kwargs["conv_in_kernel"] - 1) // 2,
            bias=True,
        )

        assert kwargs["time_embedding_type"] == "positional"
        timestep_input_dim = kwargs["block_out_channels"][0]
        time_embed_dim = kwargs["block_out_channels"][0] * 4

        self.time_proj = Timesteps(
            kwargs["block_out_channels"][0],
            kwargs["flip_sin_to_cos"],
            kwargs["freq_shift"],
        )

        kwargs["act_fn"] = getattr(hidet.graph.ops, kwargs["act_fn"])
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=kwargs["act_fn"],
        )

        if not all(
            x is None
            for x in (
                kwargs["encoder_hid_dim_type"],
                kwargs["class_embed_type"],
                kwargs["addition_embed_type"],
            )
        ):
            raise NotImplementedError(
                "Additional projection and embedding features not included yet."
            )

        self.down_blocks = []
        self.up_blocks = []

        down_block_types = kwargs["down_block_types"]
        only_cross_attention = kwargs["only_cross_attention"]
        mid_block_only_cross_attention = kwargs["mid_block_only_cross_attention"]

        if isinstance(kwargs["only_cross_attention"], bool):
            if kwargs["mid_block_only_cross_attention"] is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)  # 4

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        attention_head_dim = kwargs["attention_head_dim"]
        if isinstance(kwargs["attention_head_dim"], int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        num_attention_heads = kwargs["num_attention_heads"] or attention_head_dim
        if isinstance(kwargs["num_attention_heads"], int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        cross_attention_dim = kwargs["cross_attention_dim"]
        if isinstance(kwargs["cross_attention_dim"], int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        layers_per_block = kwargs["layers_per_block"]
        if isinstance(kwargs["layers_per_block"], int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        transformer_layers_per_block = kwargs["transformer_layers_per_block"]
        if isinstance(kwargs["transformer_layers_per_block"], int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                down_block_types
            )

        blocks_time_embed_dim = time_embed_dim
        if kwargs["class_embeddings_concat"]:
            blocks_time_embed_dim *= 2

        output_channels = kwargs["block_out_channels"][0]
        for i, down_block_type in enumerate(kwargs["down_block_types"]):
            input_channels = output_channels
            output_channels = kwargs["block_out_channels"][i]
            is_final = i == len(kwargs["block_out_channels"]) - 1

            # deleteme
            if down_block_type == "DownBlock2D":
                continue

            self.down_blocks.append(
                self.get_down_block(
                    down_block_type,
                    num_layers=layers_per_block[i],
                    input_channels=input_channels,
                    output_channels=output_channels,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    resnet_eps=kwargs["norm_eps"],
                    resnet_act_fn=kwargs["act_fn"],
                    resnet_groups=kwargs["norm_num_groups"],
                    cross_attention_dim=cross_attention_dim[i],
                    num_attention_heads=num_attention_heads[i],
                    only_cross_attention=only_cross_attention[i],
                    attention_head_dim=attention_head_dim[i] or output_channels,
                )
            )

        # fix set attr in nn Module
        self.down_blocks = nn.ModuleList(self.down_blocks)

        print(self.down_blocks)

    def get_down_block(self, down_block_type: str, **kwargs):
        if down_block_type == "CrossAttnDownBlock2D":
            return CrossAttnDownBlock2D(**{**kwargs, **self.config})  # type: ignore
        elif down_block_type == "DownBlock2D":
            return

        raise ValueError(f"{down_block_type} not found.")

    def forward(self, sample: Tensor, timesteps: Tensor) -> UNet2DConditionOutput:
        print("forward")
        if self.config["center_input_sample"]:
            sample = 2 * sample - 1.0

        timesteps = broadcast(timesteps, shape=(sample.shape[0],))
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        print(emb)
        # return UNet2DConditionOutput()


# {'sample_size': 96, 'in_channels': 4, 'out_channels': 4, 'center_input_sample': False, 'flip_sin_to_cos': True, 'freq_shift': 0, 'down_block_types': ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'], 'mid_block_type': 'UNetMidBlock2DCrossAttn', 'up_block_types': ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'], 'only_cross_attention': False, 'block_out_channels': [320, 640, 1280, 1280], 'layers_per_block': 2, 'downsample_padding': 1, 'mid_block_scale_factor': 1, 'dropout': 0.0, 'act_fn': 'silu', 'norm_num_groups': 32, 'norm_eps': 1e-05, 'cross_attention_dim': 1024, 'transformer_layers_per_block': 1, 'reverse_transformer_layers_per_block': None, 'encoder_hid_dim': None, 'encoder_hid_dim_type': None, 'attention_head_dim': [5, 10, 20, 20], 'num_attention_heads': None, 'dual_cross_attention': False, 'use_linear_projection': True, 'class_embed_type': None, 'addition_embed_type': None, 'addition_time_embed_dim': None, 'num_class_embeds': None, 'upcast_attention': True, 'resnet_time_scale_shift': 'default', 'resnet_skip_time_act': False, 'resnet_out_scale_factor': 1.0, 'time_embedding_type': 'positional', 'time_embedding_dim': None, 'time_embedding_act_fn': None, 'timestep_post_act': None, 'time_cond_proj_dim': None, 'conv_in_kernel': 3, 'conv_out_kernel': 3, 'projection_class_embeddings_input_dim': None, 'attention_type': 'default', 'class_embeddings_concat': False, 'mid_block_only_cross_attention': None, 'cross_attention_norm': None, 'addition_embed_type_num_heads': 64, '_use_default_values': ['mid_block_only_cross_attention', 'transformer_layers_per_block', 'resnet_time_scale_shift', 'resnet_out_scale_factor', 'timestep_post_act', 'class_embed_type', 'time_embedding_dim', 'conv_out_kernel', 'projection_class_embeddings_input_dim', 'attention_type', 'mid_block_type', 'addition_embed_type', 'dropout', 'cross_attention_norm', 'time_embedding_type', 'conv_in_kernel', 'time_cond_proj_dim', 'encoder_hid_dim', 'time_embedding_act_fn', 'encoder_hid_dim_type', 'addition_embed_type_num_heads', 'addition_time_embed_dim', 'resnet_skip_time_act', 'num_attention_heads', 'reverse_transformer_layers_per_block', 'class_embeddings_concat'], '_class_name': 'UNet2DConditionModel', '_diffusers_version': '0.10.0.dev0', '_name_or_path': '/home/kevin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/unet'}
