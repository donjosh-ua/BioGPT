#!/usr/bin/env python3
"""
CUDA compatibility fixer for BioGPT project
"""

import subprocess
import sys
import os
from src.gpu_diagnostics import check_compatibility, generate_install_commands


def main():
    """Fix CUDA compatibility issues"""
    print("🔧 CUDA Compatibility Fixer")
    print("=" * 40)

    # Check current state
    is_compatible = check_compatibility()

    if is_compatible:
        print("✅ Everything is already working!")
        return

    # Get the correct installation command
    pytorch_cmd = generate_install_commands()

    print("\n🚨 Issues detected. Here's what we'll do:")
    print("1. Uninstall current PyTorch")
    print("2. Install compatible PyTorch version")
    print("3. Verify installation")

    response = input("\nProceed? (y/n): ")
    if response.lower() != "y":
        print("Cancelled by user")
        return

    try:
        # Step 1: Uninstall
        print("\n📦 Uninstalling current PyTorch...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "torch",
                "torchvision",
                "torchaudio",
                "-y",
            ]
        )

        # Step 2: Install correct version
        print("\n📦 Installing compatible PyTorch...")
        if "index-url" in pytorch_cmd:
            # Handle index-url installations
            cmd_parts = pytorch_cmd.split()
            subprocess.check_call([sys.executable, "-m"] + cmd_parts[1:])
        else:
            subprocess.check_call([sys.executable, "-m"] + pytorch_cmd.split())

        # Step 3: Verify
        print("\n✅ Installation complete! Verifying...")
        check_compatibility()

        print("\n🎉 CUDA setup should now be working!")
        print("You may need to restart your Python environment.")

    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("Please try running the command manually:")
        print(f"  {pytorch_cmd}")


if __name__ == "__main__":
    main()
