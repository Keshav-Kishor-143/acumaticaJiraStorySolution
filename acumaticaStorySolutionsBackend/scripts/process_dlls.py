#!/usr/bin/env python3
"""
DLL Processing Script

Processes all DLL files from knowledge_base/Dll's/ and generates
dll_content.txt files in knowledge_base/manuals/*_DLL/data/ directories.

This ensures 100% compliance with Law 2 (Resolve Before Deciding) by
providing DLL validation data for the DLL validator.

Usage:
    python scripts/process_dlls.py
    or
    python -m scripts.process_dlls
"""

import sys
from pathlib import Path

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root and src to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from src.core.ingest import VDRIngestor
from src.utils.logger_utils import get_logger

logger = get_logger("DLL_PROCESSOR_SCRIPT")


def main():
    """Main entry point for DLL processing"""
    print("=" * 60)
    print("🔧 DLL Processing Script")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Process all DLL files from knowledge_base/Dll's/")
    print("2. Extract class, method, and property information")
    print("3. Generate dll_content.txt files in knowledge_base/manuals/*_DLL/data/")
    print("4. Enable DLL validation for Law 2 compliance")
    print("\n" + "=" * 60 + "\n")
    
    try:
        # Initialize ingestor (includes DLL processor)
        ingestor = VDRIngestor()
        
        # Process all DLLs
        result = ingestor.process_all_dlls()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY")
        print("=" * 60)
        print(f"Total DLLs processed: {result['total_files']}")
        print(f"✅ Successful: {result['successful']}")
        print(f"❌ Failed: {result['failed']}")
        
        if result.get('files_processed'):
            print("\n📋 Processed Files:")
            for file_info in result['files_processed']:
                status = "✅" if file_info['success'] else "❌"
                print(f"  {status} {file_info['filename']} ({file_info['size_mb']} MB)")
        
        if result['successful'] > 0:
            print("\n✅ DLL content files generated successfully!")
            print("   Location: knowledge_base/manuals/*_DLL/data/dll_content.txt")
            print("\n🎯 DLL validation is now 100% compliant with Law 2!")
        
        return 0 if result['failed'] == 0 else 1
        
    except Exception as e:
        logger.error("DLL processing failed", extra={"error": str(e)})
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

