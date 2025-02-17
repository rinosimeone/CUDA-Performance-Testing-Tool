import torch
import time
import matplotlib.pyplot as plt
import numpy as np

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
