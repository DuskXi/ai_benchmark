import builtins
import json
import types

import torchvision
import torch.nn as nn

from config import Config


def main():
    summary()
    pass


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


def print_models():
    build_functions, network_class_list = get_build_function()
    for name, func in build_functions.items():
        print(name)
    print(network_class_list)


def load_config(config_path="config.json"):
    config = Config.load(config_path)


def summary():
    from benchmark import summary_text, plot_benchmark_results
    result = None
    with open("results/results-1-half.json", "r") as f:
        result = json.load(f)
    keys = list(result.keys())
    keys = [key for key in keys if key != "system_information"]
    print(summary_text(result, keys))
    plot_benchmark_results(result, keys)


if __name__ == '__main__':
    main()
