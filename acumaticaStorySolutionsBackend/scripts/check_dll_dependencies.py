#!/usr/bin/env python3
"""
Check DLL Processing Dependencies

This script checks if required dependencies for DLL processing are installed.

Usage:
    python scripts/check_dll_dependencies.py
    or
    python -m scripts.check_dll_dependencies
"""

import sys
import io

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("DLL Processing Dependencies Check")
print("=" * 60)

# Check pythonnet
print("\n1. Checking pythonnet (required for full DLL extraction)...")
try:
    import clr
    print("   ✅ pythonnet: INSTALLED")
    pythonnet_available = True
except ImportError:
    print("   ❌ pythonnet: NOT INSTALLED")
    print("   💡 Install with: pip install pythonnet")
    pythonnet_available = False

# Check pefile (alternative)
print("\n2. Checking pefile (alternative metadata extraction)...")
try:
    import pefile
    print("   ✅ pefile: INSTALLED")
    pefile_available = True
except ImportError:
    print("   ⚠️  pefile: NOT INSTALLED (optional)")
    print("   💡 Install with: pip install pefile")
    pefile_available = False

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if pythonnet_available:
    print("✅ Full DLL extraction available (pythonnet installed)")
    print("   DLLs will be processed with full reflection extraction")
    print("   Classes, methods, and properties will be extracted")
elif pefile_available:
    print("⚠️  Partial DLL extraction available (pefile installed)")
    print("   DLLs will be processed with basic metadata extraction")
    print("   Limited class/method extraction")
else:
    print("❌ Only basic extraction available")
    print("   DLLs will only extract namespace hints from filenames")
    print("   No classes, methods, or properties will be extracted")
    print("\n💡 RECOMMENDATION:")
    print("   Install pythonnet for full extraction:")
    print("   pip install pythonnet")

print("=" * 60)

sys.exit(0 if pythonnet_available else 1)

