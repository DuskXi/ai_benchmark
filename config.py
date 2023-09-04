import json
import os
import types

import torchvision
from torch import nn


def get_build_function():
    network_types = []
    for model in torchvision.models.__dict__.values():
        if type(model) == type and nn.Module in model.__bases__:
            network_types.append(model)
    init_func = []
    for model in torchvision.models.__dict__.values():
        if isinstance(model, types.FunctionType):
            if model.__annotations__.get('return') in network_types:
                init_func.append(model)
    return {f.__name__: f for f in init_func}, network_types


build_functions, network_class_list = get_build_function()


class ModelConfig(dict):
    def __init__(self, **kwargs):
        data_dict = {
            "model": kwargs.get("model", None),
            "prediction": kwargs.get("prediction", False),
            "training": kwargs.get("training", False),
            "pred_batch_size": kwargs.get("pred_batch_size", 1),
            "train_batch_size": kwargs.get("train_batch_size", 1),
            "input_shape": kwargs.get("input_shape", (3, 224, 224))
        }

        super(ModelConfig, self).__init__(**data_dict)
        self.model = kwargs.get("model", None)
        self.prediction = kwargs.get("prediction", False)
        self.training = kwargs.get("training", False)
        self.pred_batch_size = kwargs.get("pred_batch_size", 1)
        self.train_batch_size = kwargs.get("train_batch_size", 1)
        self.input_shape = kwargs.get("input_shape", (3, 224, 224))

        self.build_func = None
        if self.model is None or self.model not in build_functions.keys():
            self.build_func = None
        else:
            self.build_func = build_functions[self.model]


class Config(dict):
    def __init__(self, **kwargs):
        data_dict = {
            "models": kwargs.get("models", []),  # type: list[ModelConfig]
            "sample_size": kwargs.get("sample_size", 10),

        }
        for i in range(len(data_dict["models"])):
            data_dict["models"][i] = ModelConfig(**data_dict["models"][i])

        super(Config, self).__init__(**data_dict)
        self.models = kwargs.get("models", [])  # type: list[ModelConfig]
        self.sample_size = kwargs.get("sample_size", 10)

    @staticmethod
    def load(path):
        if os.path.exists(path) or os.path.isfile(path):
            try:
                with open(path, 'r', encoding="UTF-8") as f:
                    data = json.load(f)
                    return Config(**data)
            except json.JSONDecodeError:
                pass
        else:
            config = Config(**{"models": [ModelConfig(**{"model": "vgg19"})]})
            config.save(path)
            return config

    def save(self, path):
        with open(path, 'w', encoding="UTF-8") as f:
            json.dump(self, f, indent=4, ensure_ascii=False)
