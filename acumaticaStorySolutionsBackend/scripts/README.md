# Scripts Directory

This directory contains utility scripts for processing, ingestion, and maintenance tasks.

## Available Scripts

### `process_dlls.py`
Processes all DLL files from `knowledge_base/Dll's/` and generates `dll_content.txt` files in `knowledge_base/manuals/*_DLL/data/` directories.

**Usage:**
```bash
# From project root
python scripts/process_dlls.py

# Or as module
python -m scripts.process_dlls
```

**What it does:**
1. Scans `knowledge_base/Dll's/` for DLL files
2. Extracts class, method, and property information using pythonnet reflection
3. Generates `dll_content.txt` files in respective DLL directories
4. Enables DLL validation for Law 2 compliance

### `check_dll_dependencies.py`
Checks if required dependencies for DLL processing are installed.

**Usage:**
```bash
# From project root
python scripts/check_dll_dependencies.py

# Or as module
python -m scripts.check_dll_dependencies
```

**What it checks:**
- `pythonnet` - Required for full DLL reflection extraction
- `pefile` - Optional alternative for basic metadata extraction

## Directory Structure

```
scripts/
├── __init__.py
├── README.md
├── process_dlls.py          # DLL processing utility
└── check_dll_dependencies.py # Dependency checker
```

## Notes

- All scripts handle Windows Unicode encoding automatically
- Scripts import from `src.core` and `src.utils` modules
- Scripts are designed to be run from the project root directory

