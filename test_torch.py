import torch
import time
import matplotlib.pyplot as plt
import numpy as np

import requests
import json
from datetime import datetime

def fetch_gpu_reference_data():
    """Fetch GPU performance data from online database"""
    try:
        # Try to fetch from local cache first
        try:
            with open('gpu_performance_cache.json', 'r') as f:
                cache = json.load(f)
                # Check if cache is less than 24 hours old
                if datetime.now().timestamp() - cache['timestamp'] < 86400:  # 24 hours
                    print("Using cached GPU performance data")
                    return {k: float(v) if isinstance(v, (int, float, str)) and v is not None else v 
                           for k, v in cache['data']['gpus'].items()}
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # If cache miss or expired, fetch from API
        url = "https://raw.githubusercontent.com/rinosimeone/CUDA-Performance-Testing-Tool/main/gpu_performance.json"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        gpu_data = {k: float(v) if isinstance(v, (int, float, str)) and v is not None else v 
                    for k, v in data['gpus'].items()}
        
        # Cache the results
        cache_data = {
            'timestamp': datetime.now().timestamp(),
            'data': data  # Store complete data including metadata
        }
        with open('gpu_performance_cache.json', 'w') as f:
            json.dump(cache_data, f)
        
        print("Successfully fetched latest GPU performance data")
        return gpu_data
        
    except Exception as e:
        print(f"Warning: Could not fetch online GPU data ({str(e)})")
        print("Using fallback reference data")
        # Fallback to hardcoded data if fetch fails
        return {
            'RTX 4090': 0.7234,    # Latest gen, high-end
            'RTX 3090': 0.9845,    # Previous gen, high-end
            'Quadro RTX 3000': None,  # Current GPU (will be filled during benchmark)
            'RTX 2080 Ti': 1.3456, # Two gens ago, high-end
            'RTX 2070': 1.8901,    # Two gens ago, mid-range
        }

# Initialize GPU reference data
GPU_REFERENCE = fetch_gpu_reference_data()

def create_ascii_bar(value, max_value, width=50):
    bar_length = int(width * (value / max_value))
    return '█' * bar_length + ' ' * (width - bar_length)

def show_gpu_comparison(current_time, size):
    if size != 4000:  # Only show comparison for largest matrix size
        return
        
    print("\nGPU Performance Comparison (4000x4000 matrix)")
    print("=" * 80)
    
    # Update reference with current GPU's performance
    GPU_REFERENCE['Quadro RTX 3000'] = current_time
    
    # Find max time for scaling
    max_time = max(time for time in GPU_REFERENCE.values() if time is not None)
    
    # Sort GPUs by performance (ascending times = better performance)
    sorted_gpus = sorted(
        [(gpu, time) for gpu, time in GPU_REFERENCE.items() if time is not None],
        key=lambda x: x[1]
    )
    
    # Show bars
    for gpu, time in sorted_gpus:
        marker = "→ " if gpu == 'Quadro RTX 3000' else "  "
        print(f"{marker}{gpu:<15} │ {create_ascii_bar(time, max_time)} {time:.4f}s")
    
    print("=" * 80)

def run_benchmark(size=1000, iterations=100, results=None):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Warm up GPU
        torch.cuda.synchronize()
        
        # Create tensors
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # Test CPU performance
        start_time = time.time()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        cpu_time = time.time() - start_time
        print(f"\nCPU Matrix Multiplication ({size}x{size}, {iterations} iterations)")
        print(f"Total time: {cpu_time:.4f} seconds")
        print(f"Average time per iteration: {cpu_time/iterations:.4f} seconds")
        
        # Move to GPU
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        
        # Test GPU performance
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"\nGPU Matrix Multiplication ({size}x{size}, {iterations} iterations)")
        print(f"Total time: {gpu_time:.4f} seconds")
        print(f"Average time per iteration: {gpu_time/iterations:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")
        
        # CPU vs GPU ASCII visualization
        print("\nCPU vs GPU Performance")
        print("=" * 80)
        max_time = max(cpu_time, gpu_time)
        print(f"CPU │ {create_ascii_bar(cpu_time, max_time)} {cpu_time:.4f}s")
        print(f"GPU │ {create_ascii_bar(gpu_time, max_time)} {gpu_time:.4f}s")
        print("=" * 80)
        
        # Show GPU comparison for largest matrix size
        show_gpu_comparison(gpu_time, size)
        
        if results is not None:
            results['sizes'].append(size)
            results['cpu_times'].append(cpu_time/iterations)
            results['gpu_times'].append(gpu_time/iterations)
            results['speedups'].append(speedup)
            
    else:
        print("CUDA is not available. Please check your installation.")

def plot_results(results):
    sizes = results['sizes']
    
    # Performance comparison plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(sizes, results['cpu_times'], 'b-o', label='CPU')
    plt.plot(sizes, results['gpu_times'], 'r-o', label='GPU')
    plt.title('Performance Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time per Operation (s)')
    plt.legend()
    plt.grid(True)
    
    # Speedup plot
    plt.subplot(132)
    plt.plot(sizes, results['speedups'], 'g-o')
    plt.title('GPU Speedup')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Factor (x)')
    plt.grid(True)
    
    # Bar chart comparison
    plt.subplot(133)
    x = np.arange(len(sizes))
    width = 0.35
    plt.bar(x - width/2, results['cpu_times'], width, label='CPU', color='blue', alpha=0.6)
    plt.bar(x + width/2, results['gpu_times'], width, label='GPU', color='red', alpha=0.6)
    plt.title('CPU vs GPU Performance')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time per Operation (s)')
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cuda_benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    # Initialize results dictionary
    results = {
        'sizes': [],
        'cpu_times': [],
        'gpu_times': [],
        'speedups': []
    }
    
    # Run benchmark with different matrix sizes
    print("\n=== Running CUDA Benchmarks ===")
    run_benchmark(size=1000, iterations=100, results=results)  # Small matrices
    run_benchmark(size=2000, iterations=50, results=results)   # Medium matrices
    run_benchmark(size=4000, iterations=25, results=results)   # Large matrices
    
    # Plot and save results
    plot_results(results)
    print("\nBenchmark results have been saved to 'cuda_benchmark_results.png'")
