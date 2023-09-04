import gc
import json
import os
from thop import profile
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

from config import Config, ModelConfig
from timer import Timer
import tqdm
import pynvml


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def load_config(config_path="config.json"):
    config = Config.load(config_path)
    return config


def save_results(results, path, mix_precision=False, half=False, bf16=False):
    name = f"results-0.json"
    i = 0
    while name in os.listdir(path):
        i += 1
        name = f"results-{i}{'-half' if half else ''}{'-mix_precision' if mix_precision else ''}{'-bf16' if bf16 else ''}.json"
    with open(os.path.join(path, name), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def get_current_torch_gpu_usage():
    gpu_usage_info = {
        "gpu_count": torch.cuda.device_count(),
        "gpu_usage": []
    }
    for i in range(gpu_usage_info["gpu_count"]):
        usage = {
            "index": i,
            "memory_allocated": torch.cuda.memory_allocated(i),
            "memory_cached": torch.cuda.memory_cached(i),
        }
        gpu_usage_info["gpu_usage"].append(usage)
    return gpu_usage_info


def to_size_str(size: float):
    if size < 1024:
        return f"{size}B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.2f}KB"
    elif size < 1024 ** 3:
        return f"{size / 1024 ** 2:.2f}MB"
    else:
        return f"{size / 1024 ** 3:.2f}GB"


def get_system_information():
    pynvml.nvmlInit()
    system_information = {
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        "cuda_version": pynvml.nvmlSystemGetCudaDriverVersion(),
        "device_count": pynvml.nvmlDeviceGetCount(),
        "devices": []
    }
    for i in range(system_information["device_count"]):
        device = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_information = {
            "name": pynvml.nvmlDeviceGetName(device),
            "memory": pynvml.nvmlDeviceGetMemoryInfo(device),
            "temperature": pynvml.nvmlDeviceGetTemperature(device, 0),
            "power_usage": pynvml.nvmlDeviceGetPowerUsage(device),
            "utilization": pynvml.nvmlDeviceGetUtilizationRates(device),
        }
        device_information["memory"] = {
            "total": to_size_str(device_information["memory"].total),
            "free": to_size_str(device_information["memory"].free),
            "used": to_size_str(device_information["memory"].used),
        }
        device_information["utilization"] = {
            "gpu": device_information["utilization"].gpu,
            "memory": device_information["utilization"].memory,
        }
        system_information["devices"].append(device_information)

    return system_information


def get_summary_gpu_usage(gpu_infos):
    summary = {
        "memory_allocated": 0,
        "memory_cached": 0,
    }
    for gpu_info in gpu_infos:
        summary["memory_allocated"] += gpu_info['gpu_usage'][0]["memory_allocated"]
        summary["memory_cached"] += gpu_info['gpu_usage'][0]["memory_cached"]

    return {
        "memory_allocated": to_size_str(summary["memory_allocated"] / len(gpu_infos)),
        "memory_cached": to_size_str(summary["memory_cached"] / len(gpu_infos)),
    }


def plot_benchmark_results(benchmark_results, keys):
    # Extract system information
    system_information = benchmark_results["system_information"]
    device = system_information["devices"][0]

    # Lists to store data for plotting
    labels = []
    prediction_tflops = []
    training_tflops = []

    for key in keys:
        bc_result = benchmark_results[key]
        add = False
        # Extract prediction TFlops
        if bc_result["prediction"] is not None:
            add = True
            prediction_tflops.append(bc_result['prediction']['avg_tflops'])

        # Extract training TFlops
        if bc_result["training"] is not None:
            add = True
            training_tflops.append(bc_result['training']['avg_tflops'])

        if add:
            labels.append(f"{key}")

    # Plotting
    y = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    rects1 = ax.barh(y + width / 2, prediction_tflops, width, label='Prediction TFlops', color='blue')
    rects2 = ax.barh(y - width / 2, training_tflops, width, label='Training TFlops', color='red')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('TFlops')
    ax.set_title('Benchmark Results')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()

    plot_throughput(benchmark_results, keys)


def plot_throughput(benchmark_results, keys):
    # Lists to store data for plotting
    labels = []
    prediction_throughput = []
    training_throughput = []

    for key in keys:
        bc_result = benchmark_results[key]
        add = False
        # Extract prediction throughput
        if bc_result["prediction"] is not None:
            add = True
            batch_size = bc_result["pred_batch_size"]
            throughput = (1 / bc_result['prediction']['mean']) * batch_size
            prediction_throughput.append(throughput)

        # Extract training throughput
        if bc_result["training"] is not None:
            add = True
            batch_size = bc_result["train_batch_size"]
            throughput = (1 / bc_result['training']['mean']) * batch_size
            training_throughput.append(throughput)

        if add:
            labels.append(f"{key}")

    # Plotting
    y = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    rects1 = ax.barh(y - width / 2, prediction_throughput, width, label='Prediction Throughput', color='blue')
    rects2 = ax.barh(y + width / 2, training_throughput, width, label='Training Throughput', color='red')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Items per second')
    ax.set_title('Throughput Benchmark Results')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()


def summary_text(benchmark_results, keys):
    result = ""
    system_information = benchmark_results["system_information"]
    result += "System Information:\n"
    result += f"\tDriver Version: {system_information['driver_version']}\n"
    result += f"\tCUDA Version: {system_information['cuda_version']}\n"
    device = system_information["devices"][0]
    result += f"\tDevice Name: {device['name']}\n"
    result += f"\tDevice Memory: {device['memory']['total']}\n"

    for key in keys:
        bc_result = benchmark_results[key]
        size = 1
        for s in bc_result["shape"]:
            size *= s
        result += f"{key}:\n"
        result += f"\tInput Shape: {', '.join([str(x) for x in bc_result['shape']])}\n"
        if bc_result["prediction"] is not None:
            batch_size = bc_result["pred_batch_size"]
            median = np.median(bc_result["prediction"]["durations"])
            gpu_alloc = bc_result['prediction']['gpu_states'][0]['gpu_usage'][0]['memory_allocated']
            gpu_cache = bc_result['prediction']['gpu_states'][0]['gpu_usage'][0]['memory_cached']
            result += f"\tPrediction:\n"
            result += f"\t\tTFlops: {bc_result['prediction']['avg_tflops']:.2f} TFlops\n"
            result += f"\t\tAverage: {bc_result['prediction']['mean']:.4f}s/iter, {(1 / bc_result['prediction']['mean']) * batch_size:.3f} item/s, (batch: {batch_size})\n"
            result += f"\t\tMedian: {median:.4f}s/iter, {(1 / median) * batch_size:.3f} item/s\n"
            result += f"\t\tGPU Memory Allocated: {to_size_str(gpu_alloc)}, {to_size_str(gpu_alloc / batch_size)} per item /{to_size_str(int(gpu_alloc / (batch_size * size)))}\n"
            result += f"\t\tGPU Memory Cached: {to_size_str(gpu_cache)}\n"

        if bc_result["training"] is not None:
            batch_size = bc_result["train_batch_size"]
            median = np.median(bc_result["training"]["durations"])
            gpu_alloc = bc_result['training']['gpu_states'][0]['gpu_usage'][0]['memory_allocated']
            gpu_cache = bc_result['training']['gpu_states'][0]['gpu_usage'][0]['memory_cached']
            result += f"\tTraining:\n"
            result += f"\t\tTFlops: {bc_result['training']['avg_tflops']:.2f} TFlops\n"
            result += f"\t\tAverage: {bc_result['training']['mean']:.4f}s/iter, {(1 / bc_result['training']['mean']) * batch_size:.3f} item/s, (batch: {batch_size})\n"
            result += f"\t\tMedian: {median:.4f}s/iter, {(1 / median) * batch_size:.3f} item/s\n"
            result += f"\t\tGPU Memory Allocated: {to_size_str(gpu_alloc)}, {to_size_str(gpu_alloc / batch_size)} per item /{to_size_str(int(gpu_alloc / (batch_size * size)))}\n"
            result += f"\t\tGPU Memory Cached: {to_size_str(gpu_cache)}\n"

    return result


def test_prediction(model, input_tensor, timer: Timer):
    model.eval()
    with torch.no_grad():
        timer.reset().start()
        output = model(input_tensor)
        timer.stop()
    return (output,)


def test_training(model, input_tensor, timer: Timer, optimizer, criterion):
    model.train()
    timer.reset().start()
    output = model(input_tensor)
    loss = criterion(output, torch.randint(0, 1000, (input_tensor.shape[0],)).cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    timer.stop()
    return output, loss


def benchmark(config: Config, save_path="results", mix_precision=False, half=False, bf16=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    initialized_models = []
    # initialize models
    for model_config in tqdm.tqdm(config.models, desc="Initialize models"):
        if model_config.build_func is None:
            continue
        model = model_config.build_func(pretrained=True)
        initialized_models.append((model, model_config))

    # benchmark
    timer = Timer()
    benchmark_results = {
        "system_information": get_system_information(),
    }

    bar_model = tqdm.tqdm(initialized_models, desc="Models")
    with autocast(enabled=mix_precision or bf16, dtype=torch.bfloat16) if bf16 else autocast(enabled=mix_precision):
        for model, model_config in bar_model:
            model.cuda()
            if half:
                model = model.half()

            durations_prediction = []
            tflops_prediction = []
            gpu_states_prediction = []
            durations_training = []
            tflops_training = []
            gpu_states_training = []

            if model_config.prediction:
                input_tensor = torch.rand((model_config.pred_batch_size, *model_config.input_shape)).cuda()
                if half:
                    input_tensor = input_tensor.half()
                bar_model.set_description(f"Calculating FLOPs")
                flops_p, params = profile(model, inputs=(input_tensor,), verbose=False)
                flops = flops_p
                bar_model.set_description(f"Models: {model_config.model} - prediction")
                bar_batch = tqdm.tqdm(range(config.sample_size), desc="Batch")
                for i in bar_batch:
                    r = test_prediction(model, input_tensor, timer)
                    gpu_status = get_current_torch_gpu_usage()
                    durations_prediction.append(timer.duration)
                    tflops_prediction.append(flops / ((timer.duration * 1e12) if timer.duration != 0 else 1e12))
                    gpu_states_prediction.append(gpu_status)
                    time_avg = f"{sum(durations_prediction) / len(durations_prediction):.4f}s"
                    time_min = f"{min(durations_prediction):.4f}s"
                    time_max = f"{max(durations_prediction):.4f}s"
                    gpu_info = f"{gpu_status['gpu_usage'][0]['memory_allocated'] / 1024 ** 3:.2f}GB/{gpu_status['gpu_usage'][0]['memory_cached'] / 1024 ** 3:.2f}GB"
                    bar_batch.set_description(f"Prediction-Batch: {i} - avg: {time_avg} - min: {time_min} - max: {time_max}, TFLops: {tflops_prediction[-1]:.2f}, GPU Memory: {gpu_info}")
                    for obj in r:
                        del obj
                bar_batch.close()
                del input_tensor

            if model_config.training:
                input_tensor = torch.rand((model_config.train_batch_size, *model_config.input_shape), requires_grad=True).cuda()
                if half:
                    input_tensor = input_tensor.half()
                if mix_precision:
                    for p in model.parameters():
                        p.requires_grad = True
                bar_model.set_description(f"Calculating FLOPs")
                flops_t, params = profile(model, inputs=(input_tensor,), verbose=False)
                flops = flops_t * 4 + params * 7
                bar_model.set_description(f"Models: {model_config.model} - training")
                bar_batch = tqdm.tqdm(range(config.sample_size), desc="Batch")
                optimizer = torch.optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()
                for i in bar_batch:
                    r = test_training(model, input_tensor, timer, optimizer, criterion)
                    gpu_status = get_current_torch_gpu_usage()
                    durations_training.append(timer.duration)
                    tflops_training.append(flops / ((timer.duration * 1e12) if timer.duration != 0 else 1e12))
                    gpu_states_training.append(gpu_status)
                    time_avg = f"{sum(durations_training) / len(durations_training):.4f}s"
                    time_min = f"{min(durations_training):.4f}s"
                    time_max = f"{max(durations_training):.4f}s"
                    gpu_info = f"{gpu_status['gpu_usage'][0]['memory_allocated'] / 1024 ** 3:.2f}GB/{gpu_status['gpu_usage'][0]['memory_cached'] / 1024 ** 3:.2f}GB"
                    bar_batch.set_description(f"Training-Batch: {i} - avg: {time_avg} - min: {time_min} - max: {time_max}, TFLops: {tflops_training[-1]:.2f}, GPU Memory: {gpu_info}")
                    for obj in r:
                        del obj
                bar_batch.close()
                del input_tensor

            benchmark_results[model_config.model] = {
                "shape": model_config.input_shape,
                "train_batch_size": model_config.train_batch_size,
                "pred_batch_size": model_config.pred_batch_size,
                "prediction": {"mean": sum(durations_prediction) / len(durations_prediction),
                               "min": min(durations_prediction), "max": max(durations_prediction),
                               "avg_tflops": flops_p / ((sum(durations_prediction) / len(durations_prediction)) * 1e12),
                               "sample_size": len(durations_prediction),
                               "durations": durations_prediction,
                               "tflops": tflops_prediction,
                               "gpu_summary": get_summary_gpu_usage(gpu_states_prediction),
                               "gpu_states": gpu_states_prediction} if model_config.prediction else None,
                "training": {"mean": sum(durations_training) / len(durations_training),
                             "min": min(durations_training), "max": max(durations_training),
                             "avg_tflops": flops_t / ((sum(durations_training) / len(durations_training)) * 1e12),
                             "sample_size": len(durations_training),
                             "durations": durations_training,
                             "tflops": tflops_training,
                             "gpu_summary": get_summary_gpu_usage(gpu_states_training),
                             "gpu_states": gpu_states_training} if model_config.training else None,
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()

    bar_model.close()
    save_results(benchmark_results, save_path, mix_precision, half)
    print(summary_text(benchmark_results, [model_config.model for model, model_config in initialized_models]))
    plot_benchmark_results(benchmark_results, [model_config.model for model, model_config in initialized_models])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--mix_precision", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    config = load_config(args.config)
    benchmark(config, args.save_path, args.mix_precision, args.half, args.bf16)


if __name__ == "__main__":
    main(parse_args())
