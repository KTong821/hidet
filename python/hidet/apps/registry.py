import importlib
from collections import defaultdict
from dataclasses import astuple, dataclass
from enum import Enum
from typing import Dict

from transformers import PretrainedConfig


@dataclass
class RegistryEntry:
    """
    Configuration for dynamic loading of classes.

    We expect app directories to follow this file structure:

    apps/
    ├──<model_category>/
    │   ├── modeling/
    │   │   ├── <model_name>/
    │   │   │   ├── __init__.py
    │   │   └── ...
    │   ├── processing/
    │   │   ├── <processor_name>/
    │   │   │   ├── __init__.py
    │   │   └── ...
    ├──<model_category>/
        └── ...

    For example, model_category could be "image_classification", under which "resnet" is a
    model_name. The "resnet" module could contain class ResNetImageProcessor representing a
    callable for processing images.

    Use this to dynamically load pre-processors under a general naming scheme.
    """

    model_category: str
    module_name: str
    klass: str

    def __init__(self, model_category: str, module_name: str, klass: str):
        self.model_category = model_category
        self.module_name = module_name
        self.klass = klass

class ModuleType(Enum):
    MODEL = "modeling"
    PROCESSING = "processing"

class Registry:
    module_registry: Dict[ModuleType, Dict[str, RegistryEntry]] = defaultdict(dict)

    @classmethod
    def load_module(cls, config: PretrainedConfig, module_type: ModuleType = ModuleType.MODEL):
        architectures = getattr(config, "architectures")
        if not architectures:
            raise ValueError(f"Config {config.name_or_path} has no architecture.")

        # assume only 1 architecture available for now
        architecture = architectures[0]
        if architecture not in cls.module_registry[module_type]:
            raise KeyError(
                f"No {module_type.value} class with architecture {architecture} found."
                f"Registered {module_type.value} architectures: {', '.join(cls.module_registry[module_type].keys())}."
            )

        model_category, module_name, klass = astuple(cls.module_registry[module_type][architecture])

        module = importlib.import_module(f"hidet.apps.{model_category}.{module_type.value}.{module_name}")

        if klass not in dir(module):
            raise KeyError(f"No processor class named {klass} found in module {module}.")

        return getattr(module, klass)

    @classmethod
    def register(cls, module_type: ModuleType, arch: str, entry: RegistryEntry):
        cls.module_registry[module_type][arch] = entry
