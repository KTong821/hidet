import importlib
from dataclasses import astuple, dataclass
from typing import Dict, Generic, List, Set, Type, TypeVar

import torch
from hidet.graph import Tensor, nn
from hidet.graph.nn.module import R
from transformers import PretrainedConfig

import hidet


@dataclass
class ModelRegistryEntry:
    """
    Configuration for dynamic loading of models.

    We expect app directories to follow this file structure:

    apps/
    ├──<model_category>/
    │   ├── modeling/
    │   │   ├── <model_name>/
    │   │   │   ├── __init__.py
    │   │   └── ...
    ├──<model_category>/
        └── ...

    For example, model_category could be "llm", under which "llama" is a model_name. The "llama" module
    should contain class LlamaForCausalLM representing a specific head attached to a base model.

    Use this to dynamically load models under a general naming scheme.
    """

    model_category: str
    model_name: str
    model_class: str

    def __init__(self, model_category: str, model_name: str, model_class: str):
        self.model_category = model_category
        self.model_name = model_name
        self.model_class = model_class


class PretrainedModel(nn.Module[R], Generic[R]):
    model_registry: Dict[str, ModelRegistryEntry] = {}

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def copy_weights(cls, torch_model: torch.nn.Module, hidet_model: nn.Module):
        found_tensors: List[Tensor] = []
        for name, tensor in torch_model.named_parameters():
            member = hidet_model
            for m_name in name.split("."):
                member = getattr(member, m_name)

            if not isinstance(member, Tensor):
                raise ValueError(
                    'PyTorch model "{}" defined a parameter "{}" that is not in the hidet model'.format(
                        torch_model.__class__.__name__, name
                    )
                )

            src = hidet.from_torch(tensor).to(member.dtype, member.device)
            if src.shape != member.shape:
                raise ValueError(
                    f"Parameter {name} shape mismatch, hidet: {member.shape}, torch: {src.shape}"
                )
            found_tensors.append(member)
            member.copy_(src)

        buffer_names: Set[str] = set(name for name, _ in torch_model.named_buffers())
        for name, tensor in hidet_model.named_parameters():
            if tensor not in found_tensors and name not in buffer_names:
                raise ValueError(
                    f"Parameter {name} in hidet model does not find equivalent in PyTorch model."
                )

    @classmethod
    def load_model_class(cls, config: PretrainedConfig) -> Type["PretrainedModel"]:
        architectures = getattr(config, "architectures")
        if not architectures:
            raise ValueError(f"Config {config.name_or_path} has no architecture.")

        # assume only 1 architecture available for now
        architecture = architectures[0]
        if architecture not in cls.model_registry:
            raise KeyError(
                f"No model with architecture {architecture} found."
                f"Registered architectures: {', '.join(cls.model_registry.keys())}."
            )

        model_category, model_name, model_class = astuple(
            cls.model_registry[architecture]
        )
        module = importlib.import_module(
            f"hidet.apps.{model_category}.modeling.{model_name}"
        )
        if model_class not in dir(module):
            raise KeyError(
                f"No model class named {model_class} found in module {module}."
            )

        return getattr(module, model_class)

    @classmethod
    def register_model(cls, arch: str, entry: ModelRegistryEntry):
        cls.model_registry[arch] = entry

    @classmethod
    def parse_dtype(cls, config: PretrainedConfig, default: str = "float16"):
        if config.torch_dtype:
            assert isinstance(config.torch_dtype, torch.dtype)
            return str(config.torch_dtype).split(".")[-1]
        else:
            return default
