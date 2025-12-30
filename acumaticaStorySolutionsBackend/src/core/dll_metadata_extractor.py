#!/usr/bin/env python3
"""
Alternative DLL Metadata Extractor (without pythonnet)

Uses pefile library to extract basic metadata from .NET DLL files
when pythonnet is not available. This provides better extraction
than filename-based metadata hints.
"""

import struct
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

from src.utils.logger_utils import get_logger

logger = get_logger("DLL_METADATA_EXTRACTOR")


class DLLMetadataExtractor:
    """
    Extract metadata from .NET DLL files using pefile (no pythonnet required)
    
    This is a fallback when pythonnet is not available.
    Extracts basic information from DLL metadata tables.
    """
    
    def __init__(self):
        self.logger = logger
        self.pefile_available = False
        
        try:
            import pefile
            self.pefile_available = True
            self.pefile = pefile
            self.logger.info("pefile metadata extractor initialized")
        except ImportError:
            self.logger.warning("pefile not available. Install with: pip install pefile")
    
    def extract_metadata(self, dll_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from DLL using pefile
        
        Args:
            dll_path: Path to DLL file
            
        Returns:
            Dictionary with extracted metadata
        """
        if not self.pefile_available:
            return self._extract_basic_info(dll_path)
        
        try:
            pe = self.pefile.PE(str(dll_path))
            
            extracted = {
                "dll_name": dll_path.stem,
                "assembly_name": dll_path.stem,
                "version": self._extract_version(pe),
                "namespaces": {},
                "classes": [],
                "methods": [],
                "properties": [],
                "processing_method": "pefile_metadata"
            }
            
            # Extract .NET metadata
            # Look for .NET metadata directory
            if hasattr(pe, 'DIRECTORY_ENTRY_COM_DESCRIPTOR'):
                # Try to extract from COM descriptor
                self._extract_com_metadata(pe, extracted)
            
            # Extract from string resources
            self._extract_from_strings(pe, extracted)
            
            return extracted
            
        except Exception as e:
            self.logger.error("pefile extraction failed", extra={"error": str(e)})
            return self._extract_basic_info(dll_path)
    
    def _extract_version(self, pe) -> str:
        """Extract version from PE file"""
        try:
            if hasattr(pe, 'VS_VERSIONINFO'):
                for vs in pe.VS_VERSIONINFO:
                    if hasattr(vs, 'StringFileInfo'):
                        for string_table in vs.StringFileInfo:
                            if hasattr(string_table, 'StringTable'):
                                for entry in string_table.StringTable:
                                    if hasattr(entry, 'entries'):
                                        for key, value in entry.entries.items():
                                            if 'version' in key.lower():
                                                return value
        except Exception:
            pass
        return "Unknown"
    
    def _extract_com_metadata(self, pe, extracted: Dict[str, Any]):
        """Extract metadata from COM descriptor"""
        try:
            # This is a simplified extraction
            # Full .NET metadata parsing requires parsing the metadata tables
            pass
        except Exception as e:
            self.logger.debug("COM metadata extraction failed", extra={"error": str(e)})
    
    def _extract_from_strings(self, pe, extracted: Dict[str, Any]):
        """Extract class/method names from string resources"""
        try:
            # Extract readable strings that might be class/method names
            strings = []
            seen_strings = set()
            
            # Get all sections - focus on .text and .rdata sections which contain metadata
            for section in pe.sections:
                try:
                    section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                    data = section.get_data()
                    
                    # Look for UTF-8 strings (class names, namespaces)
                    # Pattern 1: PascalCase class names: [A-Z][a-zA-Z0-9]+
                    pascal_matches = re.findall(rb'([A-Z][a-zA-Z0-9]{2,})', data)
                    for match in pascal_matches[:200]:  # Increased limit
                        try:
                            string = match.decode('utf-8', errors='ignore')
                            if len(string) >= 3 and string.isalnum() and string not in seen_strings:
                                strings.append(string)
                                seen_strings.add(string)
                        except Exception:
                            continue
                    
                    # Pattern 2: Namespace.Class patterns: [A-Za-z][A-Za-z0-9]*\.[A-Z][A-Za-z0-9]+
                    namespace_matches = re.findall(rb'([A-Za-z][A-Za-z0-9]*\.[A-Z][A-Za-z0-9]+)', data)
                    for match in namespace_matches[:200]:
                        try:
                            string = match.decode('utf-8', errors='ignore')
                            if 3 <= len(string) <= 200 and string not in seen_strings:
                                strings.append(string)
                                seen_strings.add(string)
                        except Exception:
                            continue
                except Exception as section_error:
                    self.logger.debug(f"Failed to process section {section_name}", extra={"error": str(section_error)})
                    continue
            
            # Deduplicate and categorize
            unique_strings = list(set(strings))
            self.logger.debug(f"Found {len(unique_strings)} unique strings from binary data")
            
            # Try to identify namespaces and classes (strings with dots)
            classes_added = 0
            for s in unique_strings:
                if '.' in s and 3 <= s.count('.') <= 5:  # Namespace.Class or Namespace.SubNamespace.Class
                    parts = s.split('.')
                    # Last part should be PascalCase (class name)
                    if len(parts) >= 2 and parts[-1][0].isupper():
                        namespace = '.'.join(parts[:-1])
                        class_name = parts[-1]
                        
                        # Filter out common non-class strings
                        if class_name.lower() in ['dll', 'exe', 'assembly', 'version', 'culture', 'publickeytoken']:
                            continue
                        
                        if namespace not in extracted["namespaces"]:
                            extracted["namespaces"][namespace] = []
                        
                        # Check if we already added this class
                        if not any(c['full_name'] == s for c in extracted["classes"]):
                            class_info = {
                                "name": class_name,
                                "full_name": s,
                                "namespace": namespace,
                                "base_type": None,
                                "is_abstract": False,
                                "is_public": True,
                                "is_class": True,
                                "is_interface": False,
                                "interfaces": [],
                                "custom_attributes": []
                            }
                            extracted["namespaces"][namespace].append(class_info)
                            extracted["classes"].append(class_info)
                            classes_added += 1
            
            # Also add standalone PascalCase strings as potential classes (if no namespace found)
            for s in unique_strings:
                if '.' not in s and s[0].isupper() and len(s) >= 3 and s.isalnum():
                    # Check if it's not already added
                    if not any(c['name'] == s for c in extracted["classes"]):
                        namespace = "Global"
                        if namespace not in extracted["namespaces"]:
                            extracted["namespaces"][namespace] = []
                        
                        class_info = {
                            "name": s,
                            "full_name": s,
                            "namespace": namespace,
                            "base_type": None,
                            "is_abstract": False,
                            "is_public": True,
                            "is_class": True,
                            "is_interface": False,
                            "interfaces": [],
                            "custom_attributes": []
                        }
                        extracted["namespaces"][namespace].append(class_info)
                        extracted["classes"].append(class_info)
                        classes_added += 1
            
            self.logger.info("Extracted metadata from strings", extra={
                "classes_found": len(extracted["classes"]),
                "classes_added": classes_added,
                "namespaces_found": len(extracted["namespaces"]),
                "unique_strings": len(unique_strings)
            })
            
        except Exception as e:
            self.logger.error("String extraction failed", extra={"error": str(e), "error_type": type(e).__name__})
    
    def _extract_basic_info(self, dll_path: Path) -> Dict[str, Any]:
        """Fallback: Extract basic info from filename"""
        extracted = {
            "dll_name": dll_path.stem,
            "assembly_name": dll_path.stem,
            "version": "Unknown",
            "namespaces": {},
            "classes": [],
            "methods": [],
            "properties": [],
            "processing_method": "filename_only"
        }
        
        # Extract namespace hint from filename
        if "." in dll_path.stem:
            parts = dll_path.stem.split(".")
            if len(parts) >= 2:
                namespace_hint = ".".join(parts[:-1])
                extracted["namespaces"][namespace_hint] = []
        
        return extracted

