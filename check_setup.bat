@echo off
echo ===== Checking CUDA Setup =====
echo.

echo 1. Checking NVIDIA Driver:
nvidia-smi > driver_info.txt 2>&1
type driver_info.txt
echo.

echo 2. Checking CUDA Version:
nvcc --version > cuda_info.txt 2>&1
type cuda_info.txt
echo.

echo 3. Checking Python Version:
python --version > python_info.txt 2>&1
type python_info.txt
echo.

echo 4. Checking PyTorch Installation:
python -c "import torch; print(f'PyTorch version: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}')" > pytorch_info.txt 2>&1
type pytorch_info.txt
echo.

echo 5. Running CUDA Test:
python test_torch.py > test_results.txt 2>&1
type test_results.txt
echo.

echo ===== Test Complete =====
echo Check the .txt files in this directory for detailed output
pause
