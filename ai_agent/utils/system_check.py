import os
import psutil
import torch
import subprocess
import platform
from typing import Tuple, List

def check_system_requirements(model_size_gb: float = 5.0) -> Tuple[bool, List[str]]:
    """Check if system meets minimum requirements"""
    issues = []
    
    # Check available RAM with lower threshold for CPU-only
    available_ram = psutil.virtual_memory().available / (1024**3)  # Convert to GB
    min_ram = model_size_gb * 0.5  # Reduce RAM requirement for CPU-only operation
    if available_ram < min_ram:
        issues.append(f"Very low RAM. Available: {available_ram:.1f}GB, Minimum recommended: {min_ram:.1f}GB")
    
    # Check page file size
    total_page_file = psutil.swap_memory().total / (1024**3)  # Convert to GB
    min_page_file = model_size_gb  # Reduce page file requirement
    if total_page_file < min_page_file:
        issues.append(f"Small page file. Size: {total_page_file:.1f}GB, Recommended: {min_page_file:.1f}GB")
    
    # Add hardware check
    has_gpu = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
    if not has_gpu:
        issues.append("No GPU detected - using CPU-only mode (slower performance)")
    
    # Return True if only the GPU warning is present
    critical_issues = [i for i in issues if "GPU" not in i]
    return len(critical_issues) == 0, issues

def increase_page_file_instructions() -> str:
    """Get OS-specific instructions for increasing page file"""
    if platform.system() == "Windows":
        return """To increase page file size on Windows:
1. Open System Properties (Win + Pause/Break)
2. Click Advanced system settings
3. Under Performance, click Settings
4. Click Advanced tab
5. Under Virtual memory, click Change
6. Uncheck "Automatically manage"
7. Set custom size (recommended: 16GB)"""
    else:
        return """To increase swap space on Linux:
1. Run: sudo fallocate -l 16G /swapfile
2. Run: sudo chmod 600 /swapfile
3. Run: sudo mkswap /swapfile
4. Run: sudo swapon /swapfile"""
