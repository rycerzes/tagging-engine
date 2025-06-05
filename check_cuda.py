import torch
import sys
import subprocess

def check_cuda_availability():
    """Check CUDA availability and GPU information"""
    print("=== CUDA & GPU Information ===")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if torch.cuda.is_available():
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")

    
    print("\n=== nvidia-smi Output ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi command failed or not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi not available or timed out")
    
    # Check PyTorch installation
    print("\n=== PyTorch Information ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    # Test simple CUDA operation
    if cuda_available:
        print("\n=== CUDA Test ===")
        try:
            device = torch.device("cuda")
            x = torch.randn(1000, 1000, device=device)
            y = torch.mm(x, x.t())
            print("✅ CUDA tensor operations working correctly")
        except Exception as e:
            print(f"❌ CUDA test failed: {e}")

if __name__ == "__main__":
    check_cuda_availability()
