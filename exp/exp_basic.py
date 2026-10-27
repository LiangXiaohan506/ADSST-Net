import os
import time
import numpy as np
import torch
import inspect
from models import (
    EEGNet,
    ADformer,
    Medformer,
    Conformer,
    STCGRU,
    TSLANet,
    ADSSTNet,
    MedGNN,
    CodimNet
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "EEGNet": EEGNet,
            "ADformer": ADformer,
            "Medformer": Medformer,
            "Conformer": Conformer,
            "STCGRU": STCGRU,
            "TSLANet": TSLANet,
            "ADSSTNet": ADSSTNet,
            "MedGNN": MedGNN,
            "CodimNet": CodimNet,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}\n".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU\n")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    @staticmethod
    def _model_forward_for_inference(model, x, mask):
        """统一的纯推理 forward。对支持 model_state 的模型使用 model_state='test'，避免 ADSSTNet 在测试时额外计算 contrastive/projection heads。"""
        try:
            sig = inspect.signature(model.forward)
            has_model_state = "model_state" in sig.parameters
        except (TypeError, ValueError):
            has_model_state = False
        if has_model_state:
            return model(x, mask, None, None, model_state="test")
        return model(x, mask, None, None)

    def _get_model_stats(self, model, latency_batch_size=1, throughput_batch_size=32,
                         memory_batch_size=1, warmup=100, latency_repeat=500, throughput_repeat=200):
        """
        模型统计：Total / Trainable parameters、Parameter storage / model storage、
        Peak GPU memory、Additional inference GPU memory、Inference latency (B=1)、Throughput。
        不统计 MACs/FLOPs：ADSSTNet 包含 FFT、einsum、动态图和 selective scan 等自定义算子，THOP 无法保证完整统计。
        """
        device = self.device
        was_training = model.training

        # 1. Parameters。参数占用按真实 dtype 计算，不再假设一定是 FP32；buffers 含 register_buffer() 的 A0 等。
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        parameter_storage_mb = parameter_bytes / (1024 ** 2)
        buffer_storage_mb = buffer_bytes / (1024 ** 2)
        model_storage_mb = (parameter_bytes + buffer_bytes) / (1024 ** 2)

        gpu_memory_peak_mb = gpu_memory_extra_mb = None
        latency_median_ms = latency_mean_ms = latency_std_ms = latency_p95_ms = None
        throughput = None

        try:
            model.to(device)
            model.eval()
            seq_len, enc_in = self.args.seq_len, self.args.enc_in

            # 2. GPU Memory
            if device.type == "cuda":
                x_mem = torch.zeros(memory_batch_size, seq_len, enc_in, device=device, dtype=torch.float32)
                mask_mem = torch.ones(memory_batch_size, seq_len, device=device, dtype=torch.float32)
                # 先 warm-up，避免首次 CUDA kernel 初始化干扰
                with torch.inference_mode():
                    for _ in range(20):
                        self._model_forward_for_inference(model, x_mem, mask_mem)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # 释放 PyTorch cache 中没有被使用的显存
                torch.cuda.synchronize()
                # 当前已经实际 allocated 的显存：包括 model、input 以及该进程其他仍存活 CUDA tensor
                baseline_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                with torch.inference_mode():
                    _ = self._model_forward_for_inference(model, x_mem, mask_mem)
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated(device)
                gpu_memory_peak_mb = peak_memory / (1024 ** 2)
                # forward 相对于 baseline 额外产生的 peak memory
                gpu_memory_extra_mb = max(0, peak_memory - baseline_memory) / (1024 ** 2)
                del x_mem, mask_mem
                torch.cuda.empty_cache()

            # 3. Inference latency。推荐 B=1。CUDA 使用 Event，不使用 perf_counter + 每轮 synchronize。
            x_latency = torch.zeros(latency_batch_size, seq_len, enc_in, device=device, dtype=torch.float32)
            mask_latency = torch.ones(latency_batch_size, seq_len, device=device, dtype=torch.float32)
            with torch.inference_mode():
                for _ in range(warmup):
                    self._model_forward_for_inference(model, x_latency, mask_latency)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    starts = [torch.cuda.Event(enable_timing=True) for _ in range(latency_repeat)]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(latency_repeat)]
                    for i in range(latency_repeat):
                        starts[i].record()
                        self._model_forward_for_inference(model, x_latency, mask_latency)
                        ends[i].record()
                    torch.cuda.synchronize()  # 只在全部完成后同步一次
                    latencies = np.asarray(
                        [starts[i].elapsed_time(ends[i]) for i in range(latency_repeat)], dtype=np.float64
                    )
                else:
                    latencies = []
                    for _ in range(latency_repeat):
                        t0 = time.perf_counter()
                        self._model_forward_for_inference(model, x_latency, mask_latency)
                        latencies.append((time.perf_counter() - t0) * 1000.0)
                    latencies = np.asarray(latencies, dtype=np.float64)
            latency_median_ms = float(np.median(latencies))
            latency_mean_ms = float(np.mean(latencies))
            latency_std_ms = float(np.std(latencies, ddof=1))
            latency_p95_ms = float(np.percentile(latencies, 95))
            del x_latency, mask_latency

            # 4. Throughput。不再由 latency 倒算。使用连续批量 inference，只在开始/结束同步。
            x_tp = torch.zeros(throughput_batch_size, seq_len, enc_in, device=device, dtype=torch.float32)
            mask_tp = torch.ones(throughput_batch_size, seq_len, device=device, dtype=torch.float32)
            with torch.inference_mode():
                for _ in range(warmup):
                    self._model_forward_for_inference(model, x_tp, mask_tp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(throughput_repeat):
                    self._model_forward_for_inference(model, x_tp, mask_tp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
            total_samples = throughput_batch_size * throughput_repeat
            throughput = total_samples / elapsed if elapsed > 0 else None
            del x_tp, mask_tp
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Model profiling warning] {type(e).__name__}: {e}")
        finally:
            model.train(was_training)  # 恢复进入统计前的模型状态

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "parameter_storage_mb": parameter_storage_mb,
            "buffer_storage_mb": buffer_storage_mb,
            "model_storage_mb": model_storage_mb,
            "gpu_memory_peak_mb": gpu_memory_peak_mb,
            "gpu_memory_extra_mb": gpu_memory_extra_mb,
            "latency_batch_size": latency_batch_size,
            "latency_median_ms": latency_median_ms,
            "latency_mean_ms": latency_mean_ms,
            "latency_std_ms": latency_std_ms,
            "latency_p95_ms": latency_p95_ms,
            "throughput_batch_size": throughput_batch_size,
            "throughput": throughput,
        }

    def _save_init_params_txt(self, model, save_path):
        """保存模型参数、存储大小、推理峰值显存、inference latency 和 independently measured throughput。"""
        stats = self._get_model_stats(model)
        header_lines = [
            f"# Model: {self.args.model} | Saved at init",
            f"# Input: seq_len={self.args.seq_len}, enc_in={self.args.enc_in}, num_class={self.args.num_class}",
            "#" + "=" * 72,
            f"Total parameters:\t{stats['total_params']:,}",
            f"Trainable parameters:\t{stats['trainable_params']:,}",
            f"Parameter storage:\t{stats['parameter_storage_mb']:.4f} MB",
            f"Buffer storage:\t\t{stats['buffer_storage_mb']:.4f} MB",
            f"Model storage:\t\t{stats['model_storage_mb']:.4f} MB",
        ]
        if stats["gpu_memory_peak_mb"] is not None:
            header_lines.append(f"Peak GPU memory (B=1):\t{stats['gpu_memory_peak_mb']:.4f} MB")
            header_lines.append(f"Additional inference GPU memory (B=1):\t{stats['gpu_memory_extra_mb']:.4f} MB")
        else:
            header_lines.append("GPU memory:\tN/A (CUDA unavailable)")
        if stats["latency_median_ms"] is not None:
            B = stats["latency_batch_size"]
            header_lines.append(f"Inference latency (B={B}, median):\t{stats['latency_median_ms']:.4f} ms")
            header_lines.append(
                f"Inference latency (mean ± SD):\t{stats['latency_mean_ms']:.4f} ± {stats['latency_std_ms']:.4f} ms"
            )
            header_lines.append(f"Inference latency (P95):\t{stats['latency_p95_ms']:.4f} ms")
        else:
            header_lines.append("Inference latency:\tN/A")
        if stats["throughput"] is not None:
            B = stats["throughput_batch_size"]
            header_lines.append(f"Throughput (B={B}):\t\t{stats['throughput']:.2f} samples/s")
        else:
            header_lines.append("Throughput:\t\tN/A")
        header_lines.extend(["#" + "=" * 72, ""])

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines) + "\n")
            f.write(f"A0:\t{getattr(self.args, 'A0', 'N/A')}\n\n")
            for name, param in model.named_parameters():
                p = param.detach().cpu().float()
                shape, numel, flat = tuple(p.shape), p.numel(), p.flatten().numpy()
                f.write(f"[{name}]\n")
                f.write(f"  shape: {shape}  numel: {numel}\n")
                if numel > 0:
                    f.write(f"  min: {flat.min():.8e}  max: {flat.max():.8e}  mean: {flat.mean():.8e}\n")
                    for i in range(0, numel, 10):
                        chunk = flat[i:i + 10]
                        f.write("  " + " ".join(f"{x:.8e}" for x in chunk) + "\n")
                f.write("\n")

        print("Model statistics (saved to init_params.txt):")
        for line in header_lines:
            if line:
                print("  " + line)
        print(f"  ... + per-parameter details -> {save_path}")
