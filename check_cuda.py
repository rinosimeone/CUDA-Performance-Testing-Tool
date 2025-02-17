#!/usr/bin/env python3
"""
CUDA Testing and Benchmarking Tool
This script provides comprehensive testing and benchmarking of CUDA capabilities
on the system, including hardware detection, PyTorch CUDA support, and performance testing.
"""

import subprocess
import sys
import os
import time
import platform
import torch
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# Configuration constants
MATRIX_SIZES = [1024, 2048, 4096]  # Different sizes for comprehensive testing
BENCHMARK_ITERATIONS = 10

@dataclass
class GPUBenchmark:
    name: str
    architecture: str
    type: str  # "consumer" or "professional"
    matrix_mult_1024: float  # ms
    matrix_mult_2048: float  # ms
    matrix_mult_4096: float  # ms
    memory_bandwidth: float  # GB/s
    compute_capability: str

# Database of known GPU performances (representative samples)
GPU_DATABASE = {
    # Consumer GPUs
    "RTX 4090": GPUBenchmark(
        "RTX 4090", "Ada Lovelace", "consumer",
        1.2, 3.5, 12.0, 1008, "8.9"
    ),
    "RTX 3090": GPUBenchmark(
        "RTX 3090", "Ampere", "consumer",
        1.5, 4.2, 14.5, 936, "8.6"
    ),
    "RTX 3080": GPUBenchmark(
        "RTX 3080", "Ampere", "consumer",
        1.8, 5.0, 16.0, 760, "8.6"
    ),
    "RTX 2080 Ti": GPUBenchmark(
        "RTX 2080 Ti", "Turing", "consumer",
        2.2, 6.5, 20.0, 616, "7.5"
    ),
    
    # Professional GPUs
    "RTX A6000": GPUBenchmark(
        "RTX A6000", "Ampere", "professional",
        1.4, 4.0, 13.5, 768, "8.6"
    ),
    "Quadro RTX 6000": GPUBenchmark(
        "Quadro RTX 6000", "Turing", "professional",
        2.0, 6.0, 18.5, 624, "7.5"
    ),
    "Quadro RTX 4000": GPUBenchmark(
        "Quadro RTX 4000", "Turing", "professional",
        2.5, 7.5, 24.0, 416, "7.5"
    ),
    "Quadro RTX 3000": GPUBenchmark(
        "Quadro RTX 3000", "Turing", "professional",
        0.8, 5.6, 36.6, 256, "7.5"
    ),
    "RTX 4080": GPUBenchmark(
        "RTX 4080", "Ada Lovelace", "consumer",
        1.4, 4.0, 13.5, 717, "8.9"
    ),
    "RTX 3070": GPUBenchmark(
        "RTX 3070", "Ampere", "consumer",
        2.0, 6.0, 19.0, 448, "8.6"
    ),
    "Quadro RTX 5000": GPUBenchmark(
        "Quadro RTX 5000", "Turing", "professional",
        1.8, 5.5, 17.5, 448, "7.5"
    ),
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_nvidia_smi() -> bool:
    """
    Check NVIDIA GPU and driver status using nvidia-smi.
    
    Returns:
        bool: True if nvidia-smi executed successfully, False otherwise
    """
    try:
        logger.info("Esecuzione di 'nvidia-smi' per verificare scheda video e driver NVIDIA:")
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except FileNotFoundError:
        logger.error("nvidia-smi non trovato: verifica di aver installato i driver NVIDIA e di aver aggiunto nvidia-smi al PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Errore nell'esecuzione di nvidia-smi: {e.stderr}")
        return False

def check_pytorch_cuda() -> bool:
    """
    Check PyTorch CUDA availability and configuration.
    
    Returns:
        bool: True if CUDA is available in PyTorch, False otherwise
    """
    try:
        has_cuda = torch.cuda.is_available()
        logger.info("\nVerifica disponibilità CUDA in PyTorch:")
        
        if has_cuda:
            logger.info("  -> CUDA è disponibile in PyTorch.")
            logger.info(f"  -> Nome GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  -> Numero di GPU disponibili: {torch.cuda.device_count()}")
            return True
        else:
            logger.warning("  -> CUDA non disponibile o driver non compatibili.")
            return False
            
    except ImportError:
        logger.error("\nPyTorch non è installato. Se vuoi testare CUDA da PyTorch, installalo con:\n  pip install torch\n")
        return False

def check_cpu() -> Tuple[str, int]:
    """
    Get CPU information.
    
    Returns:
        Tuple[str, int]: CPU model name and number of cores
    """
    cpu = platform.processor() or "Informazione non disponibile"
    cores = os.cpu_count() or 0
    logger.info("Informazioni CPU:")
    logger.info(f"  -> Processore: {cpu}")
    logger.info(f"  -> Numero di core: {cores}\n")
    return cpu, cores

def check_cuda_capability() -> Tuple[bool, str, float, str]:
    """
    Check CUDA capabilities of the system.
    
    Returns:
        Tuple[bool, str, float, str]: (success, device_name, total_memory, compute_capability)
    """
    logger.info("Verifica disponibilità CUDA:")
    if torch.cuda.is_available():
        try:
            device_index = 0
            device_name = torch.cuda.get_device_name(device_index)
            props = torch.cuda.get_device_properties(device_index)
            total_memory = props.total_memory / (1024**3)
            compute_capability = f"{props.major}.{props.minor}"
            
            logger.info("  -> CUDA è disponibile in PyTorch.")
            logger.info(f"  -> Nome GPU: {device_name}")
            logger.info(f"  -> Memoria totale: {total_memory:.2f} GB")
            logger.info(f"  -> Compute Capability: {compute_capability}\n")
            
            return True, device_name, total_memory, compute_capability
        except Exception as e:
            logger.error(f"Errore durante il controllo delle capacità CUDA: {e}")
            return False, "", 0.0, ""
    else:
        logger.warning("  -> CUDA non disponibile o driver non compatibili.\n")
        return False, "", 0.0, ""

def measure_memory_bandwidth() -> float:
    """
    Measure memory bandwidth using a simple memory copy test.
    
    Returns:
        float: Memory bandwidth in GB/s
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        # Allocate large tensors (2GB)
        size = 2 * 1024 * 1024 * 1024 // 4  # 2GB in float32
        a = torch.randn(size, device='cuda')
        b = torch.empty_like(a)
        
        # Warm-up
        b.copy_(a)
        torch.cuda.synchronize()
        
        # Measure
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        b.copy_(a)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        bandwidth = (size * 4 * 2) / (elapsed_ms / 1000) / 1e9  # GB/s
        
        del a, b
        torch.cuda.empty_cache()
        
        return bandwidth
    except Exception as e:
        logger.error(f"Errore durante la misurazione della banda di memoria: {e}")
        return 0.0

def run_cuda_benchmark(size: int, n_iter: int = BENCHMARK_ITERATIONS) -> Optional[float]:
    """
    Run CUDA benchmark using matrix multiplication.
    
    Args:
        size: Size of the square matrices to multiply
        n_iter: Number of iterations for the benchmark
        
    Returns:
        Optional[float]: Average time in milliseconds if successful, None otherwise
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.device("cuda")
        # Crea due matrici casuali sulla GPU
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # Riscaldamento (warm-up)
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
        
        times: List[float] = []
        for _ in range(n_iter):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = torch.mm(A, B)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms)
        
        # Pulizia memoria GPU
        del A, B
        torch.cuda.empty_cache()
        
        avg_time = sum(times) / len(times)
        logger.info(f"Benchmark matrice {size}x{size}: {avg_time:.2f} ms")
        return avg_time
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione del benchmark: {e}")
        return None

def create_bar_chart(values: List[float], labels: List[str], title: str, max_width: int = 50) -> str:
    """
    Create an ASCII bar chart.
    
    Args:
        values: List of values to plot
        labels: List of labels for each value
        title: Chart title
        max_width: Maximum width of bars in characters
        
    Returns:
        str: Formatted ASCII bar chart
    """
    if not values:
        return ""
    
    # Find maximum value and label length
    max_val = max(values)
    max_label_len = max(len(label) for label in labels)
    
    # Create chart
    chart = f"\n{title}\n"
    chart += "=" * (max_label_len + max_width + 15) + "\n"
    
    # Create bars
    for value, label in zip(values, labels):
        # Calculate bar width
        width = int((value / max_val) * max_width) if max_val != 0 else 0
        bar = "█" * width
        
        # Add value at the end of the bar
        chart += f"{label:<{max_label_len}} │ {bar:<{max_width}} {value:.1f}\n"
    
    return chart

def format_comparison_table(current_results: Dict[str, float], gpu_name: str) -> str:
    """
    Format benchmark results as a comparison table.
    
    Args:
        current_results: Dictionary with current benchmark results
        gpu_name: Name of the current GPU
        
    Returns:
        str: Formatted comparison table
    """
    # Header
    table = "\nComparazione Performance GPU:\n"
    table += "-" * 100 + "\n"
    table += f"{'GPU Name':<20} {'Type':<12} {'Architecture':<12} "
    table += f"{'1024x1024':<12} {'2048x2048':<12} {'4096x4096':<12} "
    table += f"{'Memory GB/s':<12} {'Compute Cap':<10}\n"
    table += "-" * 100 + "\n"
    
    # Current GPU results
    table += f"{gpu_name:<20} {'Current':<12} {'---':<12} "
    table += f"{current_results['matrix_1024']:<12.2f} "
    table += f"{current_results['matrix_2048']:<12.2f} "
    table += f"{current_results['matrix_4096']:<12.2f} "
    table += f"{current_results['memory_bandwidth']:<12.2f} "
    table += f"{current_results['compute_capability']:<10}\n"
    
    # Reference GPUs
    for gpu in GPU_DATABASE.values():
        table += f"{gpu.name:<20} {gpu.type:<12} {gpu.architecture:<12} "
        table += f"{gpu.matrix_mult_1024:<12.2f} {gpu.matrix_mult_2048:<12.2f} "
        table += f"{gpu.matrix_mult_4096:<12.2f} {gpu.memory_bandwidth:<12.2f} "
        table += f"{gpu.compute_capability:<10}\n"
    
    # Create performance visualization charts
    # 1. Matrix multiplication comparison for current size
    matrix_chart_data = {
        '1024x1024': [],
        '2048x2048': [],
        '4096x4096': []
    }
    
    # Collect data for charts
    matrix_labels = []
    for gpu_label, gpu in [("Current", None)] + list(GPU_DATABASE.items()):
        matrix_labels.append(gpu_label if gpu else "Current")
        if gpu:
            matrix_chart_data['1024x1024'].append(gpu.matrix_mult_1024)
            matrix_chart_data['2048x2048'].append(gpu.matrix_mult_2048)
            matrix_chart_data['4096x4096'].append(gpu.matrix_mult_4096)
        else:
            matrix_chart_data['1024x1024'].append(current_results['matrix_1024'])
            matrix_chart_data['2048x2048'].append(current_results['matrix_2048'])
            matrix_chart_data['4096x4096'].append(current_results['matrix_4096'])
    
    table += "\nVisualizzazione Performance (tempo minore = migliore):\n"
    for size in ['1024x1024', '2048x2048', '4096x4096']:
        table += create_bar_chart(
            matrix_chart_data[size][:5],  # Show only top 5 for clarity
            matrix_labels[:5],
            f"Performance Moltiplicazione Matrice {size} (ms)"
        )
    
    # Memory bandwidth comparison
    bandwidth_values = [current_results['memory_bandwidth']]
    bandwidth_labels = ["Current"]
    for name, gpu in list(GPU_DATABASE.items())[:4]:  # Top 4 GPUs
        bandwidth_values.append(gpu.memory_bandwidth)
        bandwidth_labels.append(name)
    
    table += create_bar_chart(
        bandwidth_values,
        bandwidth_labels,
        "Banda Memoria (GB/s) - maggiore = migliore"
    )
    
    # Performance scaling chart with theoretical vs actual scaling
    scaling_values = [1.0]  # Normalized to 1024x1024 performance
    scaling_labels = ["1024x1024 (base)"]
    base_perf = current_results['matrix_1024']
    
    for size in [2048, 4096]:
        actual_scaling = current_results[f'matrix_{size}'] / base_perf if base_perf != 0 else 0
        # Theoretical scaling is (size/1024)^3 for matrix multiplication
        theoretical_scaling = (size / 1024)**3
        efficiency = (theoretical_scaling / actual_scaling) * 100 if actual_scaling != 0 else 0
        
        scaling_values.append(actual_scaling)
        scaling_labels.append(f"{size}x{size} ({efficiency:.0f}% efficienza)")
    
    table += create_bar_chart(
        scaling_values,
        scaling_labels,
        "Scaling Performance (rispetto a 1024x1024 come base)\n"
        "L'efficienza indica quanto ci si avvicina allo scaling teorico ottimale"
    )
    
    return table

def _get_relative_performance(current: float, reference: float) -> float:
    """
    Helper function to get the relative performance difference.
    Positive result means 'current' is faster (less time) than 'reference'.
    """
    if reference == 0:
        return 0.0
    return (reference - current) / reference * 100

def _find_similar_gpus(current_results: Dict[str, float], gpu_name: str) -> str:
    """
    Identify similar GPUs based on matrix multiplication times and memory bandwidth.
    Returns a formatted string describing the similar GPUs.
    """
    similar_report = ""
    similar_gpus = []
    for gpu in GPU_DATABASE.values():
        # Calculate similarity score based on multiple metrics
        matrix_diff = abs(gpu.matrix_mult_2048 - current_results['matrix_2048'])
        if gpu.memory_bandwidth == 0:
            memory_diff = 1.0
        else:
            memory_diff = abs(gpu.memory_bandwidth - current_results['memory_bandwidth']) / gpu.memory_bandwidth
        small_matrix_diff = abs(gpu.matrix_mult_1024 - current_results['matrix_1024'])
        large_matrix_diff = abs(gpu.matrix_mult_4096 - current_results['matrix_4096'])
        
        # Weighted similarity score
        similarity_score = (
            matrix_diff * 0.4 +  # Medium matrix weight
            memory_diff * 0.3 +  # Memory bandwidth weight
            small_matrix_diff * 0.15 +  # Small matrix weight
            large_matrix_diff * 0.15    # Large matrix weight
        )
        
        if similarity_score < 3.0:  # Adjusted threshold for better matching
            similar_gpus.append((gpu, similarity_score))
    
    similar_gpus.sort(key=lambda x: x[1])
    if similar_gpus:
        similar_report += f"\n1. La tua GPU ({gpu_name}) ha performance simili a:\n"
        for gpu, score in similar_gpus[:3]:
            # Calculate relative performance for detailed comparison
            matrix_perf = _get_relative_performance(current_results['matrix_2048'], gpu.matrix_mult_2048)
            memory_perf = _get_relative_performance(gpu.memory_bandwidth, current_results['memory_bandwidth'])
            
            comparison = "simile a"
            if matrix_perf > 0 and abs(matrix_perf) >= 10:
                comparison = "più veloce di"
            elif matrix_perf < 0 and abs(matrix_perf) >= 10:
                comparison = "più lenta di"
            similar_report += f"   - {gpu.name} ({gpu.type}, {gpu.architecture})\n"
            similar_report += f"     • Performance calcolo: {comparison} ({abs(matrix_perf):.1f}%)\n"
            if abs(memory_perf) > 10:
                similar_report += f"     • Banda memoria: {abs(memory_perf):.1f}% inferiore\n"
    return similar_report

def _categorize_performance(current_results: Dict[str, float], gpu_name: str) -> str:
    """
    Determine a performance category for the current GPU based on reference GPU data.
    Returns a formatted string with the category analysis.
    """
    analysis = ""
    matrix_2048_time = current_results['matrix_2048']
    matrix_4096_time = current_results['matrix_4096']
    memory_bandwidth = current_results['memory_bandwidth']
    compute_capability = current_results['compute_capability']

    # Separate consumer and professional GPUs for fairer comparison
    prof_gpus = [gpu for gpu in GPU_DATABASE.values() if gpu.type == "professional"]
    same_gen_gpus = [gpu for gpu in prof_gpus if gpu.compute_capability == compute_capability]

    # Calculate scores relative to same-generation professional GPUs
    if same_gen_gpus:
        avg_prof_2048 = sum(gpu.matrix_mult_2048 for gpu in same_gen_gpus) / len(same_gen_gpus)
        avg_prof_bandwidth = sum(gpu.memory_bandwidth for gpu in same_gen_gpus) / len(same_gen_gpus) if len(same_gen_gpus) > 0 else 1
        relative_compute = avg_prof_2048 / matrix_2048_time if matrix_2048_time != 0 else 0
        relative_bandwidth = memory_bandwidth / avg_prof_bandwidth if avg_prof_bandwidth != 0 else 0

        score = (relative_compute * 60 + relative_bandwidth * 40)  # Weighted score

        if score > 1.2:
            category = "fascia alta (professionale)"
        elif score > 0.8:
            category = "fascia media-alta (professionale)"
        elif score > 0.6:
            category = "fascia media (professionale)"
        else:
            category = "fascia base (professionale)"
    else:
        # Fallback if no same-generation professional GPUs found
        if matrix_2048_time < 5.0:
            category = "fascia alta"
        elif matrix_2048_time < 7.0:
            category = "fascia media-alta"
        elif matrix_2048_time < 10.0:
            category = "fascia media"
        else:
            category = "fascia base"

    analysis += f"\n2. Categoria di performance: {category}\n"
    return analysis

def _analyze_detailed_performance(current_results: Dict[str, float]) -> str:
    """
    Build a detailed strengths/weaknesses analysis based on matrix performance and memory bandwidth.
    Returns the formatted analysis string.
    """
    analysis = "\n3. Analisi dettagliata:\n"
    matrix_2048_time = current_results['matrix_2048']
    matrix_4096_time = current_results['matrix_4096']
    memory_bandwidth = current_results['memory_bandwidth']
    compute_capability = current_results['compute_capability']

    # Determine professional GPU subset for same-compute
    prof_gpus = [gpu for gpu in GPU_DATABASE.values() if gpu.type == "professional"]
    same_gen_gpus = [gpu for gpu in prof_gpus if gpu.compute_capability == compute_capability]

    # Memory bandwidth analysis relative to same class
    if same_gen_gpus:
        filtered_gpus = [gpu for gpu in same_gen_gpus if gpu.memory_bandwidth > 0]
        if not filtered_gpus:
            analysis += f"   ℹ Banda di memoria: {memory_bandwidth:.0f} GB/s\n"
        else:
            prof_bandwidth_percentile = sum(1 for gpu in filtered_gpus if gpu.memory_bandwidth < memory_bandwidth) / len(filtered_gpus) * 100
            analysis += f"   ℹ Banda di memoria: {memory_bandwidth:.0f} GB/s "
            if prof_bandwidth_percentile > 50:
                analysis += f"(superiore al {prof_bandwidth_percentile:.0f}% delle GPU professionali {filtered_gpus[0].architecture})\n"
            else:
                analysis += f"(nella media delle GPU professionali {filtered_gpus[0].architecture})\n"
    else:
        analysis += f"   ℹ Banda di memoria: {memory_bandwidth:.0f} GB/s\n"

    # Matrix multiplication analysis by size, comparing to same-generation professional GPUs
    analysis += "\n   Performance per dimensione matrice:\n"
    sizes = [(1024, 'piccole'), (2048, 'medie'), (4096, 'grandi')]
    for size, desc in sizes:
        current_time = current_results[f'matrix_{size}']
        if same_gen_gpus:
            times = [getattr(gpu, f'matrix_mult_{size}') for gpu in same_gen_gpus]
            avg_time = sum(times) / len(times) if len(times) > 0 else 1
            perf_ratio = (avg_time / current_time - 1) * 100 if current_time != 0 else 0

            if abs(perf_ratio) < 10:
                analysis += f"   ✓ Matrici {desc} ({size}x{size}): {current_time:.1f}ms "
                analysis += f"(nella media delle GPU professionali {filtered_gpus[0].architecture if same_gen_gpus else ''})\n"
            elif perf_ratio > 0:
                analysis += f"   ✓ Matrici {desc} ({size}x{size}): {current_time:.1f}ms "
                analysis += f"({perf_ratio:.0f}% più veloce della media)\n"
            else:
                analysis += f"   ℹ Matrici {desc} ({size}x{size}): {current_time:.1f}ms "
                analysis += f"({-perf_ratio:.0f}% più lenta della media)\n"
        else:
            analysis += f"   ℹ Matrici {desc} ({size}x{size}): {current_time:.1f}ms\n"

    return analysis

def _build_workload_recommendations(current_results: Dict[str, float]) -> str:
    """
    Build a final set of recommendations based on the performance profile.
    """
    recs = "\n4. Raccomandazioni per il carico di lavoro:\n"
    small_matrix_perf = current_results['matrix_1024']
    large_matrix_perf = current_results['matrix_4096']
    memory_bandwidth = current_results['memory_bandwidth']

    if small_matrix_perf < 1.0:  # Excellent small matrix performance
        recs += "   ✓ Ottimo per:\n"
        recs += "     • Operazioni batch con matrici piccole (<2048x2048)\n"
        recs += "     • Inferenza di reti neurali con batch size ridotti\n"
        recs += "     • Elaborazione real-time di dati\n"

    if large_matrix_perf > 30.0:  # Slower on large matrices
        recs += "   ⚠ Considerazioni per carichi pesanti:\n"
        recs += "     • Suddividere operazioni su matrici grandi in blocchi più piccoli\n"
        recs += "     • Ottimizzare l'uso della memoria per dataset estesi\n"
        recs += "     • Considerare tecniche di parallelizzazione per task complessi\n"

    if memory_bandwidth < 300:
        recs += "\n   Suggerimenti per l'ottimizzazione della memoria:\n"
        recs += "     • Minimizzare i trasferimenti di dati tra CPU e GPU\n"
        recs += "     • Riutilizzare i dati in memoria GPU quando possibile\n"
        recs += "     • Considerare la compressione dei dati per dataset grandi\n"

    recs += "\n   Suggerimenti per CUDA 12.x:\n"
    recs += "     • Utilizzare Tensor Cores per operazioni in FP16/BF16 quando possibile\n"
    recs += "     • Abilitare TF32 per migliori performance in FP32\n"
    recs += "     • Sfruttare le ottimizzazioni della libreria cuBLAS\n"
    recs += "     • Considerare l'uso di CUDA Graphs per workload ripetitivi\n"

    return recs

def analyze_performance(current_results: Dict[str, float], gpu_name: str) -> str:
    """
    Analyze performance compared to known GPUs.
    
    Args:
        current_results: Dictionary with current benchmark results
        gpu_name: Name of the current GPU
        
    Returns:
        str: Performance analysis text
    """
    # Start analysis string
    analysis = "\nAnalisi Performance:\n"

    # 1. Find GPUs with similar performance
    analysis += _find_similar_gpus(current_results, gpu_name)

    # 2. Categorize performance
    analysis += _categorize_performance(current_results, gpu_name)

    # 3. Detailed analysis by memory bandwidth and matrix size
    analysis += _analyze_detailed_performance(current_results)

    # 4. Build final workload recommendations
    analysis += _build_workload_recommendations(current_results)

    return analysis

def main() -> None:
    """
    Main function to run all CUDA tests and benchmarks.
    """
    try:
        if not check_nvidia_smi():
            logger.warning("Test interrotto: nvidia-smi non disponibile")
            return
            
        if not check_pytorch_cuda():
            logger.warning("Test interrotto: PyTorch CUDA non disponibile")
            return
            
        check_cpu()
        cuda_available, device_name, total_memory, compute_capability = check_cuda_capability()
        
        if cuda_available:
            logger.info("\nEsecuzione benchmark completo...")

            # Run benchmarks
            current_results = {
                'matrix_1024': run_cuda_benchmark(1024) or 0.0,
                'matrix_2048': run_cuda_benchmark(2048) or 0.0,
                'matrix_4096': run_cuda_benchmark(4096) or 0.0,
                'memory_bandwidth': measure_memory_bandwidth(),
                'compute_capability': compute_capability
            }

            # Show comparison table
            print(format_comparison_table(current_results, device_name))

            # Show performance analysis
            print(analyze_performance(current_results, device_name))
            
        else:
            logger.warning("Impossibile eseguire il benchmark senza una GPU CUDA disponibile.")
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione dei test: {e}")
    finally:
        # Manteniamo aperta la finestra terminale
        input("\nPremi Invio per terminare...")

if __name__ == "__main__":
    main()
