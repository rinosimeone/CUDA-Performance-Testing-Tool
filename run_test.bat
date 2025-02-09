@echo off
python test_torch.py > test_output.txt 2>&1
type test_output.txt
pause
