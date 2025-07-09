"""
GPU and CUDA compatibility diagnostic tool
"""

import subprocess
import sys
import logging
import re
from packaging import version

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_nvidia_driver_version():
    """Get NVIDIA driver version"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract driver version from nvidia-smi output
            match = re.search(r"Driver Version: (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
        return None
    except FileNotFoundError:
        return None


def get_cuda_version():
    """Get CUDA toolkit version"""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract CUDA version
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
        return None
    except FileNotFoundError:
        return None


def get_pytorch_info():
    """Get PyTorch version and CUDA support info"""
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.enabled else None
            ),
        }
    except ImportError:
        return None


def check_compatibility():
    """Check compatibility between all components"""
    logger.info("🔍 Checking GPU/CUDA compatibility...")

    # Get versions
    driver_version = get_nvidia_driver_version()
    cuda_version = get_cuda_version()
    pytorch_info = get_pytorch_info()

    logger.info("📊 System Information:")
    logger.info(f"  NVIDIA Driver: {driver_version or 'Not found'}")
    logger.info(f"  CUDA Toolkit: {cuda_version or 'Not found'}")

    if pytorch_info:
        logger.info(f"  PyTorch: {pytorch_info['version']}")
        logger.info(
            f"  PyTorch CUDA: {pytorch_info['cuda_version'] or 'Not available'}"
        )
        logger.info(f"  CUDA Available: {pytorch_info['cuda_available']}")
    else:
        logger.info("  PyTorch: Not installed")

    # Compatibility matrix for CUDA versions and PyTorch
    compatibility_matrix = {
        # PyTorch version: [supported CUDA versions]
        "2.0": ["11.7", "11.8"],
        "2.1": ["11.8", "12.1"],
        "2.2": ["11.8", "12.1"],
        "2.3": ["11.8", "12.1"],
        "2.4": ["11.8", "12.1", "12.4"],
        "2.5": ["11.8", "12.1", "12.4"],
        "2.6": ["11.8", "12.1", "12.4"],
        "2.7": ["11.8", "12.1", "12.4"],
    }

    # Driver compatibility (minimum driver version for CUDA)
    driver_cuda_compatibility = {
        "11.7": "515.43",
        "11.8": "520.61",
        "12.1": "530.30",
        "12.4": "550.54",
    }

    issues = []
    recommendations = []

    # Check if components are installed
    if not driver_version:
        issues.append("❌ NVIDIA drivers not found")
        recommendations.append(
            "Install NVIDIA drivers from https://www.nvidia.com/drivers"
        )

    if not cuda_version:
        issues.append("❌ CUDA toolkit not found")
        recommendations.append(
            "Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
        )

    if not pytorch_info:
        issues.append("❌ PyTorch not installed")
        recommendations.append("Install PyTorch with CUDA support")
    elif not pytorch_info["cuda_available"]:
        issues.append("❌ PyTorch CUDA not available")
        recommendations.append("Reinstall PyTorch with CUDA support")

    # Check version compatibility
    if pytorch_info and pytorch_info["cuda_version"] and cuda_version:
        pytorch_major_minor = ".".join(pytorch_info["version"].split(".")[:2])
        pytorch_cuda = pytorch_info["cuda_version"]

        if pytorch_major_minor in compatibility_matrix:
            supported_cuda = compatibility_matrix[pytorch_major_minor]
            if pytorch_cuda not in supported_cuda:
                issues.append(
                    f"❌ PyTorch {pytorch_info['version']} expects CUDA {pytorch_cuda}, but CUDA {cuda_version} is installed"
                )
                recommendations.append(
                    f"Install PyTorch for CUDA {cuda_version} or install CUDA {pytorch_cuda}"
                )

    # Check driver compatibility
    if driver_version and cuda_version and cuda_version in driver_cuda_compatibility:
        min_driver = driver_cuda_compatibility[cuda_version]
        if version.parse(driver_version) < version.parse(min_driver):
            issues.append(
                f"❌ NVIDIA driver {driver_version} is too old for CUDA {cuda_version} (minimum: {min_driver})"
            )
            recommendations.append(
                f"Update NVIDIA drivers to version {min_driver} or higher"
            )

    # Print results
    if not issues:
        logger.info("✅ All components are compatible!")
    else:
        logger.info("\n🚨 Compatibility Issues Found:")
        for issue in issues:
            logger.info(f"  {issue}")

        logger.info("\n💡 Recommendations:")
        for rec in recommendations:
            logger.info(f"  • {rec}")

    return len(issues) == 0


def generate_install_commands():
    """Generate appropriate installation commands"""
    logger.info("\n🛠️  Installation Commands:")

    driver_version = get_nvidia_driver_version()
    cuda_version = get_cuda_version()

    # Determine best PyTorch installation
    if cuda_version:
        cuda_major_minor = ".".join(cuda_version.split(".")[:2])

        if cuda_major_minor in ["11.8"]:
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        elif cuda_major_minor in ["12.1"]:
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_major_minor in ["12.4"]:
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        else:
            pytorch_cmd = "pip install torch torchvision torchaudio"
            logger.warning(f"No PyTorch CUDA build available for CUDA {cuda_version}")
    else:
        pytorch_cmd = "pip install torch torchvision torchaudio"

    logger.info(f"PyTorch installation: {pytorch_cmd}")

    if not driver_version:
        logger.info(
            "NVIDIA Drivers: Visit https://www.nvidia.com/drivers and download for your GPU"
        )

    if not cuda_version:
        logger.info("CUDA Toolkit: Visit https://developer.nvidia.com/cuda-downloads")

    return pytorch_cmd


def main():
    """Main diagnostic function"""
    logger.info("🚀 GPU/CUDA Compatibility Checker")
    logger.info("=" * 50)

    is_compatible = check_compatibility()
    pytorch_cmd = generate_install_commands()

    if not is_compatible:
        logger.info("\n📋 Quick Fix Steps:")
        logger.info("1. Update NVIDIA drivers if needed")
        logger.info("2. Install/update CUDA toolkit if needed")
        logger.info(f"3. Reinstall PyTorch: {pytorch_cmd}")
        logger.info("4. Restart your system")
        logger.info("5. Run this script again to verify")


if __name__ == "__main__":
    main()
