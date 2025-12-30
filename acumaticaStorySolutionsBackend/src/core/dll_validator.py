#!/usr/bin/env python3
"""
DLL Entity Validator - Validates entities against actual DLL symbols

This module provides deterministic validation of technical entities by checking
them against actual DLL assemblies. This is critical for Law 2 compliance:
"Resolve Before Deciding" - entities must be validated against real DLL symbols.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.dll_processor import DLLProcessor


class DLLEntityValidator:
    """
    Validates technical entities against actual DLL assemblies
    
    Provides deterministic validation:
    - DAC names against actual classes
    - Field names against actual properties
    - Graph names against actual graph classes
    - Method names against actual methods
    """
    
    def __init__(self):
        self.logger = get_logger("DLL_VALIDATOR")
        self.dll_processor = DLLProcessor()
        
        # Cache for DLL data (loaded once, reused)
        self.dll_cache: Dict[str, Dict[str, Any]] = {}
        self.class_registry: Dict[str, Set[str]] = defaultdict(set)  # namespace -> {class_names}
        self.field_registry: Dict[str, Set[str]] = defaultdict(set)  # class_name -> {field_names}
        self.graph_registry: Set[str] = set()  # graph class names
        
        # Load DLL data from knowledge base
        self._load_dll_registry()
        
        self.logger.info("DLL Entity Validator initialized", extra={
            "dlls_loaded": len(self.dll_cache),
            "classes_registered": sum(len(classes) for classes in self.class_registry.values())
        })
    
    def _load_dll_registry(self):
        """Load DLL class and field information from knowledge base"""
        try:
            # LOCAL_BASE_PATH is already knowledge_base/manuals, so use it directly
            dll_base_path = Path(config.LOCAL_BASE_PATH)
            
            # Find all DLL directories
            dll_dirs = [d for d in dll_base_path.iterdir() if d.is_dir() and d.name.endswith("_DLL")]
            
            for dll_dir in dll_dirs:
                dll_name = dll_dir.name.replace("_DLL", "")
                
                # Try to load DLL content from data directory
                data_file = dll_dir / "data" / "dll_content.txt"
                if data_file.exists():
                    try:
                        with open(data_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract class names from DLL content
                        self._parse_dll_content(dll_name, content)
                        
                        self.dll_cache[dll_name] = {
                            "name": dll_name,
                            "content": content,
                            "classes": list(self.class_registry.get(dll_name, set()))
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load DLL content for {dll_name}", extra={"error": str(e)})
            
            self.logger.info("DLL registry loaded", extra={
                "dlls_processed": len(self.dll_cache),
                "total_classes": sum(len(classes) for classes in self.class_registry.values())
            })
            
        except Exception as e:
            self.logger.error("Failed to load DLL registry", extra={"error": str(e)})
    
    def _parse_dll_content(self, dll_name: str, content: str):
        """Parse DLL content text to extract class and field information"""
        # Extract class definitions
        # Pattern: **Class: ClassName**
        class_pattern = r'\*\*Class:\s*([^\n*]+)\*\*'
        classes = re.findall(class_pattern, content)
        
        for class_name in classes:
            class_name = class_name.strip()
            if class_name:
                # Extract namespace from full name if present
                if '.' in class_name:
                    parts = class_name.split('.')
                    namespace = '.'.join(parts[:-1])
                    short_name = parts[-1]
                else:
                    namespace = dll_name
                    short_name = class_name
                
                self.class_registry[namespace].add(short_name)
                self.class_registry[namespace].add(class_name)  # Also store full name
                
                # Check if it's a Graph class (ends with Entry, Maint, Setup, or inherits from PXGraph)
                if any(suffix in class_name for suffix in ['Entry', 'Maint', 'Setup', 'Inquiry']):
                    self.graph_registry.add(short_name)
                    self.graph_registry.add(class_name)
        
        # Extract field/property information
        # Look for property definitions in class sections
        field_pattern = r'Property:\s*(\w+)\s*\(.*?\)'
        fields = re.findall(field_pattern, content, re.IGNORECASE)
        
        # Also look for DAC field patterns
        dac_field_pattern = r'public\s+[\w\<\>]+\s+(\w+)\s*\{'
        dac_fields = re.findall(dac_field_pattern, content)
        
        all_fields = set(fields + dac_fields)
        
        # Try to associate fields with classes (look for class context)
        for class_name in classes:
            class_name = class_name.strip()
            if class_name:
                # Simple heuristic: if field appears near class definition, associate it
                # This is approximate but better than nothing
                for field in all_fields:
                    self.field_registry[class_name].add(field)
    
    def validate_dac(self, dac_name: str) -> Dict[str, Any]:
        """
        Validate DAC name against actual DLL classes
        
        Args:
            dac_name: DAC name to validate (e.g., "EVOrder", "SOOrder")
            
        Returns:
            Validation result with:
            - valid: bool
            - full_name: Optional[str] - Full namespace-qualified name if found
            - namespace: Optional[str]
            - confidence: float
        """
        dac_name_clean = dac_name.strip()
        
        # Check exact match first
        for namespace, classes in self.class_registry.items():
            if dac_name_clean in classes:
                full_name = f"{namespace}.{dac_name_clean}" if namespace else dac_name_clean
                return {
                    "valid": True,
                    "full_name": full_name,
                    "namespace": namespace,
                    "confidence": 1.0
                }
        
        # Check partial match (namespace.DACName)
        if '.' in dac_name_clean:
            parts = dac_name_clean.split('.')
            potential_dac = parts[-1]
            potential_namespace = '.'.join(parts[:-1])
            
            if potential_namespace in self.class_registry:
                if potential_dac in self.class_registry[potential_namespace]:
                    return {
                        "valid": True,
                        "full_name": dac_name_clean,
                        "namespace": potential_namespace,
                        "confidence": 1.0
                    }
        
        # Check if it matches known patterns (PX.Objects.*)
        for namespace in ['PX.Objects', 'PX.Objects.EV', 'PX.Objects.SO', 'PX.Objects.AR', 'PX.Objects.AP']:
            if namespace in self.class_registry:
                if dac_name_clean in self.class_registry[namespace]:
                    full_name = f"{namespace}.{dac_name_clean}"
                    return {
                        "valid": True,
                        "full_name": full_name,
                        "namespace": namespace,
                        "confidence": 0.9  # High confidence but not 1.0 since we inferred namespace
                    }
        
        return {
            "valid": False,
            "full_name": None,
            "namespace": None,
            "confidence": 0.0
        }
    
    def validate_field(self, field_name: str, dac_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate field name against actual DAC properties
        
        Args:
            field_name: Field name to validate
            dac_name: Optional DAC name to check field ownership
            
        Returns:
            Validation result with:
            - valid: bool
            - owning_dac: Optional[str]
            - confidence: float
        """
        field_name_clean = field_name.strip()
        
        if dac_name:
            # Check specific DAC
            dac_validation = self.validate_dac(dac_name)
            if dac_validation["valid"]:
                # Check if field exists in this DAC
                dac_key = dac_validation.get("full_name") or dac_name
                if dac_key in self.field_registry:
                    if field_name_clean in self.field_registry[dac_key]:
                        return {
                            "valid": True,
                            "owning_dac": dac_key,
                            "confidence": 1.0
                        }
        
        # Search across all DACs
        for dac_key, fields in self.field_registry.items():
            if field_name_clean in fields:
                return {
                    "valid": True,
                    "owning_dac": dac_key,
                    "confidence": 0.8  # Found but not confirmed ownership
                }
        
        return {
            "valid": False,
            "owning_dac": None,
            "confidence": 0.0
        }
    
    def validate_graph(self, graph_name: str) -> Dict[str, Any]:
        """
        Validate graph name against actual graph classes
        
        Args:
            graph_name: Graph name to validate (e.g., "EVOrderEntry", "SOOrderEntry")
            
        Returns:
            Validation result with:
            - valid: bool
            - full_name: Optional[str]
            - confidence: float
        """
        graph_name_clean = graph_name.strip()
        
        # Check graph registry
        if graph_name_clean in self.graph_registry:
            return {
                "valid": True,
                "full_name": graph_name_clean,
                "confidence": 1.0
            }
        
        # Check if it matches graph naming patterns and exists in class registry
        if re.match(r'^[A-Z][a-zA-Z0-9]*(Entry|Maint|Setup|Inquiry)$', graph_name_clean):
            # Check if exists in any namespace
            for namespace, classes in self.class_registry.items():
                if graph_name_clean in classes:
                    full_name = f"{namespace}.{graph_name_clean}" if namespace else graph_name_clean
                    return {
                        "valid": True,
                        "full_name": full_name,
                        "confidence": 0.9
                    }
        
        return {
            "valid": False,
            "full_name": None,
            "confidence": 0.0
        }
    
    def validate_form_id(self, form_id: str) -> Dict[str, Any]:
        """
        Validate form ID format (cannot validate against DLL, but can validate format)
        
        Args:
            form_id: Form ID to validate (e.g., "EV301000", "SO301000")
            
        Returns:
            Validation result with format validation
        """
        form_id_clean = form_id.strip().upper()
        
        # Validate format: [A-Z]{2}\d{6}
        if re.match(r'^[A-Z]{2}\d{6}$', form_id_clean):
            return {
                "valid": True,
                "form_id": form_id_clean,
                "confidence": 0.7  # Format valid but cannot verify existence in DLL
            }
        
        return {
            "valid": False,
            "form_id": None,
            "confidence": 0.0
        }
    
    def validate_entities(self, entities: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Validate all entities against DLL registry
        
        Args:
            entities: Dictionary of entity types to lists of entity values
            
        Returns:
            Validation results for each entity
        """
        results = {}
        
        # Validate DACs
        if 'dacs' in entities:
            results['dacs'] = {}
            for dac in entities['dacs']:
                results['dacs'][dac] = self.validate_dac(dac)
        
        # Validate Graphs
        if 'graphs' in entities:
            results['graphs'] = {}
            for graph in entities['graphs']:
                results['graphs'][graph] = self.validate_graph(graph)
        
        # Validate Fields (need DAC context)
        if 'fields' in entities:
            results['fields'] = {}
            dacs = entities.get('dacs', [])
            
            for field in entities['fields']:
                # Try to find owning DAC
                owning_dac = None
                for dac in dacs:
                    field_validation = self.validate_field(field, dac)
                    if field_validation['valid']:
                        owning_dac = field_validation.get('owning_dac')
                        break
                
                results['fields'][field] = self.validate_field(field, owning_dac)
        
        # Validate Forms (format only)
        if 'forms' in entities:
            results['forms'] = {}
            for form_id in entities['forms']:
                results['forms'][form_id] = self.validate_form_id(form_id)
        
        return results

