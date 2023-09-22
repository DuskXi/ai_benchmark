import builtins
import json
import types

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn

from config import Config


def main():
    plot_all()
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


def summary(file, flag, turbo=False):
    from benchmark import summary_text, plot_benchmark_results
    result = None
    with open(file, "r") as f:
        result = json.load(f)
    keys = list(result.keys())
    keys = [key for key in keys if key != "system_information"]
    print(summary_text(result, keys))
    plot_benchmark_results(result, keys, result["system_information"]["devices"][0]["name"] + "-" + flag + ("" if not turbo else "-Turbo"), True)


def test_data():
    result = None
    with open("results_3090/results-0.json", "r") as f:
        result = json.load(f)
    keys = list(result.keys())
    keys = [key for key in keys if key != "system_information"]
    prepare(result, keys)


def prepare(benchmark_results, keys):
    # Extract system information
    system_information = benchmark_results["system_information"]
    device = system_information["devices"][0]

    # Lists to store data for plotting
    labels = []
    prediction_tflops = []
    training_tflops = []
    prediction_throughput = []
    training_throughput = []

    for key in keys:
        bc_result = benchmark_results[key]
        add = False
        # Extract prediction TFlops
        if bc_result["prediction"] is not None:
            add = True
            prediction_tflops.append(bc_result['prediction']['avg_tflops'])
            batch_size = bc_result["pred_batch_size"]
            throughput = (1 / bc_result['prediction']['mean']) * batch_size
            prediction_throughput.append(throughput)

        # Extract training TFlops
        if bc_result["training"] is not None:
            add = True
            training_tflops.append(bc_result['training']['avg_tflops'])
            batch_size = bc_result["train_batch_size"]
            throughput = (1 / bc_result['training']['mean']) * batch_size
            training_throughput.append(throughput)

        if add:
            labels.append(f"{key}")

    return {
        "labels": labels,
        "prediction_tflops": prediction_tflops,
        "training_tflops": training_tflops,
        "prediction_throughput": prediction_throughput,
        "training_throughput": training_throughput
    }


def plot(data, title="benchmark results", save=False):
    plot_data = {}

    for key, (benchmark_results, keys) in data.items():
        plot_data[key] = prepare(benchmark_results, keys)
    print(json.dumps(plot_data, indent=4))

    labels = list(plot_data.values())[0]["labels"]
    devices = list(plot_data.keys())
    y = np.arange(len(labels)) * 1.3
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "black"]
    width = 0.35

    fig, ax = plt.subplots(figsize=(24, 16))
    plot_arg = {k: {} for k in devices}
    for i in range(len(labels)):
        base = {k: v for k, v in plot_data[devices[0]].items() if k != "labels"}
        for device in devices:
            bm_data = {k: v for k, v in plot_data[device].items() if k != "labels"}
            for index, (k, v) in enumerate(bm_data.items()):
                key = f"{device}-{k}"
                if k not in plot_arg[device]:
                    plot_arg[device][k] = {"offset": index, "array": []}
                plot_arg[device][k]["array"] = np.array(v) / np.array(base[k]) * 100

    suby = []
    unit = 1 / (5 * len(devices))
    for i, (device, bars) in enumerate(plot_arg.items()):
        rects1 = ax.barh(y + unit * 4 + unit * i, bars['prediction_throughput']['array'], unit, label=device, color=colors[i])
        rects2 = ax.barh(y + unit * 4 * 2 + unit * i, bars['training_throughput']['array'], unit, label=device, color=colors[i])
        rects3 = ax.barh(y + unit * 4 * 3 + unit * i, bars['prediction_tflops']['array'], unit, label=device, color=colors[i])
        rects4 = ax.barh(y + unit * 4 * 4 + unit * i, bars['training_tflops']['array'], unit, label=device, color=colors[i])
        ax.bar_label(rects1, padding=3, labels=[f'{v:.2f}%' for v in bars['prediction_throughput']['array']])
        ax.bar_label(rects2, padding=3, labels=[f'{v:.2f}%' for v in bars['training_throughput']['array']])
        ax.bar_label(rects3, padding=3, labels=[f'{v:.2f}%' for v in bars['prediction_tflops']['array']])
        ax.bar_label(rects4, padding=3, labels=[f'{v:.2f}%' for v in bars['training_tflops']['array']])
        if i == 0:
            suby.append(y + unit * 4 + unit * i)
            suby.append(y + unit * 4 * 2 + unit * i)
            suby.append(y + unit * 4 * 3 + unit * i)
            suby.append(y + unit * 4 * 4 + unit * i)
    #
    suby = np.concatenate(suby)
    suby.sort()
    suby = suby + ((len(devices) / 2) * unit)
    ax.set_yticks(suby)
    value = ['prediction_throughput', 'training_throughput', 'prediction_tflops', 'training_tflops'] * len(labels)
    ax.set_yticklabels(value, fontweight='bold')

    y = y + 0.7
    # ax.set_yticks(y)
    # ax.set_yticklabels(labels)
    for i, label in enumerate(labels):
        ax.text(-45, y[i], label, ha='center', va='center', fontweight='bold', fontsize='18')

    ax.legend()
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    if save:
        plt.savefig(f"{title}.png")
    plt.show()


def plot_all():
    config = {
        "full": {
            "3090-full": "results_3090/results-0.json",
            "4090-full": "results_4090/results-0.json",
            # "4090-Turbo-full": "results/results-0.json",
            "4090-Server-full": "results_4090_server/results-0.json",
            "A100-full": "results_a100/results-0.json",
        },
        "mix": {
            "3090-mix_precision": "results_3090/results-1-mix_precision.json",
            "4090-mix_precision": "results_4090/results-1-mix_precision.json",
            # "4090-Turbo-mix_precision": "results/results-0-mix_precision.json",
            "4090-Server-mix_precision": "results_4090_server/results-0-mix_precision.json",
            "A100-mix_precision": "results_a100/results-0-mix_precision.json"
        },
        "half": {
            "3090-half": "results_3090/results-1-half.json",
            "4090-half": "results_4090/results-1-half.json",
            # "4090-Turbo-half": "results/results-0-half.json",
            "4090-Server-half": "results_4090_server/results-0-half.json",
            "A100-half": "results_a100/results-0-half.json"
        }
    }
    for precision, files in config.items():
        plot_by_cfg(files, precision)


def plot_by_cfg(files, precision):
    # files = {
    #     "3090-full": "results_3090/results-0.json", "3090-mix_precision": "results_3090/results-1-mix_precision.json",
    #     "4090-full": "results_4090/results-0.json", "4090-mix_precision": "results_4090/results-1-mix_precision.json",
    #     "4090-Turbo-full": "results/results-0.json", "4090-Turbo-mix_precision": "results/results-0-mix_precision.json"
    # }
    # files = {
    #     "3090-full": "results_3090/results-0.json",
    #     "4090-full": "results_4090/results-0.json",
    #     "4090-Turbo-full": "results/results-0.json",
    # }
    # files = {
    #      "3090-mix_precision": "results_3090/results-1-mix_precision.json",
    #      "4090-mix_precision": "results_4090/results-1-mix_precision.json",
    #      "4090-Turbo-mix_precision": "results/results-0-mix_precision.json"
    # }
    # files = {
    #     "3090-half": "results_3090/results-1-half.json",
    #     "4090-half": "results_4090/results-1-half.json",
    #     "4090-Turbo-half": "results/results-0-half.json"
    # }

    data = {}

    for k, file in files.items():
        result = None
        with open(file, "r") as f:
            result = json.load(f)
        keys = list(result.keys())
        keys = [key for key in keys if key != "system_information"]
        data[k] = (result, keys)

    # set dpi as 600
    plt.rcParams["figure.dpi"] = 400
    plot(data, title=f"{precision} precision benchmark results a100 cp", save=True)


def summlist():
    files = [
        "results_3090/results-0.json", "results_3090/results-1-mix_precision.json",
        "results_4090/results-0.json", "results_4090/results-1-mix_precision.json",
    ]
    files = []
    flags = ["full", "mix_precision", "full", "mix_precision"]
    flags = ["full", "mix_precision"]
    for file, flag in zip(files, flags):
        summary(file, flag, True)


if __name__ == '__main__':
    main()
