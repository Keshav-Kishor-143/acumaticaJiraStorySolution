#!/usr/bin/env python3
"""
Business Objects Generator - Auto-generates semantic object index entries from DLL metadata

This script:
1. Reads DLL metadata from knowledge base
2. Extracts DACs, Graphs, and relationships
3. Generates business_objects.json entries
4. Merges with existing manual entries
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logger_utils import get_logger
from src.core.dll_reflection_index import DllReflectionIndexBuilder, TypeRecord
from src.config.config import config

logger = get_logger("BUSINESS_OBJECTS_GENERATOR")


class BusinessObjectsGenerator:
    """Generates business_objects.json entries from DLL metadata"""
    
    def __init__(self):
        self.knowledge_base_path = Path(config.LOCAL_BASE_PATH).parent
        self.business_objects_path = self.knowledge_base_path / "manuals" / "business_objects.json"
        self.screen_mappings_path = self.knowledge_base_path / "manuals" / "screen_id_mappings.json"
        self.dll_index_builder = DllReflectionIndexBuilder()
        self.screen_mappings = self._load_screen_mappings()
        
    def _extract_module_from_namespace(self, full_name: str) -> str:
        """Extract module name from full DAC/Graph name"""
        if not full_name:
            return ""
        
        # Extract namespace parts
        parts = full_name.split('.')
        if len(parts) >= 2:
            # Return first two parts (e.g., "PX.Objects" or "NV.Rental360")
            return '.'.join(parts[:2])
        return parts[0] if parts else ""
    
    def _load_screen_mappings(self) -> Dict[str, Any]:
        """Load screen ID mappings from file"""
        if self.screen_mappings_path.exists():
            try:
                with open(self.screen_mappings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading screen mappings: {e}")
        
        return {"mappings": {}, "screen_id_patterns": {}, "form_to_screen_mappings": {}}
    
    def _infer_screen_id(self, dac_full_name: str, dac_name: str, graph_name: str, module: str) -> str:
        """Infer screen ID from DAC/Graph name and module using mappings"""
        mappings = self.screen_mappings.get("mappings", {})
        patterns = self.screen_mappings.get("screen_id_patterns", {})
        
        # First, try exact DAC full name match
        if dac_full_name in mappings:
            return mappings[dac_full_name].get("primary_screen_id", "")
        
        # Try module-based patterns
        module_prefix = module.split('.')[-1] if '.' in module else module
        if module_prefix in patterns:
            module_patterns = patterns[module_prefix]
            
            # Check for common document types
            dac_lower = dac_name.lower()
            if "order" in dac_lower and "order" in module_patterns:
                return module_patterns["order"]
            elif "shipment" in dac_lower and "shipment" in module_patterns:
                return module_patterns.get("shipment", "")
            elif "invoice" in dac_lower and "invoice" in module_patterns:
                return module_patterns["invoice"]
            elif "receipt" in dac_lower and "receipt" in module_patterns:
                return module_patterns.get("receipt", "")
            elif "transfer" in dac_lower and "transfer" in module_patterns:
                return module_patterns.get("transfer", "")
            elif "item" in dac_lower and "item" in module_patterns:
                return module_patterns.get("item", "")
            elif "appointment" in dac_lower and "appointment" in module_patterns:
                return module_patterns.get("appointment", "")
            elif "opportunity" in dac_lower and "opportunity" in module_patterns:
                return module_patterns.get("opportunity", "")
            elif "request" in dac_lower and "request" in module_patterns:
                return module_patterns.get("request", "")
            elif "requisition" in dac_lower and "requisition" in module_patterns:
                return module_patterns.get("requisition", "")
        
        # Fallback: try DAC name patterns
        screen_id_patterns = {
            "SOOrder": "SO301000",
            "SOLine": "SO301000",
            "SOShipment": "SO302000",
            "POOrder": "PO301000",
            "POLine": "PO301000",
            "POReceipt": "PO302000",
            "ARInvoice": "AR301000",
            "ARTran": "AR301000",
            "APInvoice": "AP301000",
            "APTran": "AP301000",
            "CROpportunity": "CR304000",
            "FSAppointment": "FS300200",
            "INTransfer": "IN304000",
            "InventoryItem": "IN202500",
            "RQRequest": "RQ301000",
            "RQRequisition": "RQ303000",
            "NVRTReturnTicket": "NV304000",
            "NVRTRentalTicket": "NV303000",
            "NAWEXOrder": "NAW402000",
        }
        
        # Try DAC name first
        for key, screen_id in screen_id_patterns.items():
            if key in dac_name:
                return screen_id
        
        # Try Graph name
        for key, screen_id in screen_id_patterns.items():
            if key in graph_name:
                return screen_id
        
        return ""
    
    def _infer_key_fields(self, dac_name: str, module: str) -> Dict[str, str]:
        """Infer common key fields based on DAC name and module"""
        key_fields = {
            "document_number": "",
            "customer": "",
            "date": "",
            "status": ""
        }
        
        # Common field name patterns
        if "Order" in dac_name:
            key_fields["document_number"] = "OrderNbr"
            key_fields["date"] = "OrderDate"
        elif "Invoice" in dac_name or "AR" in dac_name or "AP" in dac_name:
            key_fields["document_number"] = "RefNbr"
            key_fields["date"] = "DocDate"
        elif "Shipment" in dac_name:
            key_fields["document_number"] = "ShipmentNbr"
            key_fields["date"] = "ShipDate"
        elif "Receipt" in dac_name:
            key_fields["document_number"] = "ReceiptNbr"
            key_fields["date"] = "ReceiptDate"
        elif "Transfer" in dac_name:
            key_fields["document_number"] = "RefNbr"
            key_fields["date"] = "TransferDate"
        elif "Ticket" in dac_name:
            key_fields["document_number"] = "RefNbr"
            key_fields["date"] = "DocDate"
        elif "Opportunity" in dac_name:
            key_fields["document_number"] = "OpportunityID"
            key_fields["date"] = "DocumentDate"
        elif "Item" in dac_name or "Inventory" in dac_name:
            key_fields["document_number"] = "InventoryID"
            key_fields["status"] = "ItemStatus"
        
        # Customer/Vendor fields
        if "SO" in module or "AR" in module or "CR" in module or "FS" in module:
            key_fields["customer"] = "CustomerID"
        elif "PO" in module or "AP" in module:
            key_fields["vendor"] = "VendorID"
        elif "CR" in module:
            key_fields["customer"] = "BAccountID"
        
        # Status field
        if not key_fields["status"]:
            key_fields["status"] = "Status"
        
        return key_fields
    
    def _find_line_dac(self, dac: TypeRecord, all_dacs: List[TypeRecord], graphs: List[TypeRecord]) -> Optional[str]:
        """Find line DAC for a header DAC"""
        dac_name = dac.name or ""
        module = self._extract_module_from_namespace(dac.full_name)
        
        # Common line DAC patterns
        line_patterns = [
            f"{dac_name}Line",
            f"{dac_name}Tran",
            f"{dac_name}Detail",
            f"{dac_name}Split",
        ]
        
        # Remove common suffixes
        base_name = re.sub(r"(Order|Invoice|Shipment|Receipt|Ticket|Request|Requisition)$", "", dac_name)
        if base_name != dac_name:
            line_patterns.extend([
                f"{base_name}Line",
                f"{base_name}Tran",
                f"{base_name}Detail",
            ])
        
        # Search for matching line DACs
        for pattern in line_patterns:
            for candidate in all_dacs:
                if candidate.name == pattern and self._extract_module_from_namespace(candidate.full_name) == module:
                    return candidate.full_name
        
        return None
    
    def _generate_aliases(self, concept: str, dac_name: str) -> List[str]:
        """Generate aliases for a concept"""
        aliases = []
        
        # Common alias patterns
        concept_lower = concept.lower()
        dac_lower = dac_name.lower()
        
        # Add variations
        if "order" in concept_lower:
            aliases.extend([f"{concept} Document", concept.replace("Order", "Order Document")])
        if "invoice" in concept_lower:
            aliases.extend([f"{concept} Document", concept.replace("Invoice", "Invoice Document")])
        if "ticket" in concept_lower:
            aliases.extend([concept.replace("Ticket", "Document"), concept.replace("Ticket", "Entry")])
        
        # Add module-specific aliases
        if "rental" in dac_lower and "return" in dac_lower:
            aliases.extend(["Return Ticket", "Equipment Return", "Return Document"])
        if "rental" in dac_lower and "ticket" in dac_lower and "return" not in dac_lower:
            aliases.extend(["Rental Order", "Rental Document"])
        
        return list(set(aliases))  # Remove duplicates
    
    def _determine_type(self, dac_name: str, cache_name: str) -> str:
        """Determine object type (document, master, template, etc.)"""
        name_lower = (dac_name + " " + cache_name).lower()
        
        if any(word in name_lower for word in ["setup", "preferences", "class", "type"]):
            return "master"
        elif any(word in name_lower for word in ["template", "email", "report"]):
            return "template"
        elif any(word in name_lower for word in ["order", "invoice", "shipment", "receipt", "ticket", "request", "requisition", "transfer"]):
            return "document"
        else:
            return "master"
    
    def _determine_priority(self, module: str, concept: str) -> int:
        """Determine priority (higher = more important)"""
        # Custom modules get higher priority
        if "NV." in module or "NAW" in module:
            return 10
        
        # Core documents get medium-high priority
        if any(word in concept.lower() for word in ["order", "invoice", "shipment", "receipt"]):
            return 8
        
        # Master data gets lower priority
        return 5
    
    def _generate_retrieval_tags(self, concept: str, module: str, dac_name: str) -> List[str]:
        """Generate retrieval tags for semantic search"""
        tags = []
        
        # Add concept words
        words = re.findall(r'\b\w+\b', concept.lower())
        tags.extend([w for w in words if len(w) > 2])
        
        # Add module tag
        if module:
            module_clean = module.replace(".", " ").lower()
            tags.append(module_clean)
        
        # Add DAC name words
        dac_words = re.findall(r'[A-Z][a-z]+', dac_name)
        tags.extend([w.lower() for w in dac_words if len(w) > 2])
        
        return list(set(tags))  # Remove duplicates
    
    def generate_from_dll(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Generate business object entries from DLL metadata"""
        logger.info("Generating business objects from DLL metadata")
        
        try:
            idx = self.dll_index_builder.load_or_build(force_rebuild=False)
            dacs = [r for r in idx.records if r.kind == "dac" and (r.cache_name or "").strip()]
            graphs = [r for r in idx.records if r.kind == "graph"]
            all_dacs = [r for r in idx.records if r.kind == "dac"]
            
            graph_by_full: Dict[str, TypeRecord] = {g.full_name: g for g in graphs if g.full_name}
            
            generated_objects = []
            
            for dac in dacs:
                cache_name = (dac.cache_name or "").strip()
                if not cache_name:
                    continue
                
                # Skip if confidence too low (based on evidence)
                confidence = 0.75  # Default confidence
                if not dac.primary_graphs:
                    confidence = 0.6  # Lower confidence if no primary graph
                
                if confidence < min_confidence:
                    continue
                
                # Get primary graph
                graph_full = ""
                if dac.primary_graphs:
                    graph_full = dac.primary_graphs[0]
                else:
                    # Try to find graph by name pattern
                    dac_name = dac.name or ""
                    graph_patterns = [
                        f"{dac_name}Entry",
                        f"{dac_name}Maint",
                        f"{dac_name}EntryMaint",
                    ]
                    for pattern in graph_patterns:
                        for graph in graphs:
                            if graph.name == pattern:
                                graph_full = graph.full_name
                                break
                        if graph_full:
                            break
                
                if not graph_full:
                    continue  # Skip if no graph found
                
                module = self._extract_module_from_namespace(dac.full_name)
                screen_id = self._infer_screen_id(dac.full_name, dac.name or "", graph_full.split('.')[-1], module)
                
                # Find line DAC
                line_dac = self._find_line_dac(dac, all_dacs, graphs)
                line_dac_full = line_dac or ""
                
                # Get line screen IDs from mappings if available
                mappings_data = self.screen_mappings.get("mappings", {})
                line_screen_ids = []
                if dac.full_name in mappings_data:
                    line_screen_ids = mappings_data[dac.full_name].get("line_screen_ids", [])
                elif screen_id:
                    line_screen_ids = [screen_id]  # Default to same screen as header
                
                # Generate entry
                entry = {
                    "concept": cache_name,
                    "aliases": self._generate_aliases(cache_name, dac.name or ""),
                    "module": module,
                    "type": self._determine_type(dac.name or "", cache_name),
                    "priority": self._determine_priority(module, cache_name),
                    "primary_dac": dac.full_name,
                    "primary_graph": graph_full,
                    "primary_screen_id": screen_id,
                    "line_dac": line_dac_full.split('.')[-1] if line_dac_full else "",
                    "line_graph_extension": "",
                    "line_screen_ids": line_screen_ids if line_screen_ids else ([screen_id] if screen_id and line_dac_full else []),
                    "key_fields": self._infer_key_fields(dac.name or "", module),
                    "custom_fields": {},
                    "relationships": {
                        "header_to_line_relation": f"{dac.name}.{self._infer_key_fields(dac.name or '', module).get('document_number', 'RefNbr')} = {line_dac_full.split('.')[-1]}.{self._infer_key_fields(dac.name or '', module).get('document_number', 'RefNbr')}" if line_dac_full else "",
                        "extensions": [f"{dac.name}Ext"],
                        "child_records": [line_dac_full.split('.')[-1]] if line_dac_full else []
                    },
                    "patterns_applicable": [
                        "validation_rule",
                        "persist_time_enforcement"
                    ],
                    "retrieval_tags": self._generate_retrieval_tags(cache_name, module, dac.name or ""),
                    "notes": f"Auto-generated from DLL metadata: {dac.full_name}"
                }
                
                generated_objects.append(entry)
            
            logger.info(f"Generated {len(generated_objects)} business object entries from DLL metadata")
            return generated_objects
            
        except Exception as e:
            logger.error(f"Error generating business objects from DLL: {e}", exc_info=True)
            return []
    
    def load_existing(self) -> Dict[str, Any]:
        """Load existing business_objects.json"""
        if self.business_objects_path.exists():
            try:
                with open(self.business_objects_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading existing business_objects.json: {e}")
        
        return {
            "version": "1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "objects": []
        }
    
    def merge_objects(self, existing: List[Dict], generated: List[Dict], 
                     merge_strategy: str = "merge") -> List[Dict]:
        """Merge existing and generated objects"""
        if merge_strategy == "replace":
            return generated
        
        # Create lookup by concept
        existing_by_concept = {obj.get("concept", ""): obj for obj in existing}
        
        merged = []
        processed_concepts = set()
        
        # Add existing objects first (manual entries take precedence)
        for obj in existing:
            concept = obj.get("concept", "")
            if concept:
                merged.append(obj)
                processed_concepts.add(concept.lower())
        
        # Add generated objects that don't conflict
        for obj in generated:
            concept = obj.get("concept", "")
            if concept and concept.lower() not in processed_concepts:
                merged.append(obj)
                processed_concepts.add(concept.lower())
        
        return merged
    
    def save(self, data: Dict[str, Any]):
        """Save business_objects.json"""
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        try:
            self.business_objects_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.business_objects_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved business_objects.json with {len(data.get('objects', []))} objects")
        except Exception as e:
            logger.error(f"Error saving business_objects.json: {e}", exc_info=True)
            raise
    
    def generate_and_merge(self, merge_strategy: str = "merge", min_confidence: float = 0.7):
        """Generate from DLL and merge with existing"""
        logger.info("Starting business objects generation and merge")
        
        # Load existing
        existing_data = self.load_existing()
        existing_objects = existing_data.get("objects", [])
        
        # Generate from DLL
        generated_objects = self.generate_from_dll(min_confidence=min_confidence)
        
        # Merge
        merged_objects = self.merge_objects(existing_objects, generated_objects, merge_strategy)
        
        # Save
        existing_data["objects"] = merged_objects
        self.save(existing_data)
        
        logger.info(f"Generation complete: {len(existing_objects)} existing + {len(generated_objects)} generated = {len(merged_objects)} total")
        
        return merged_objects


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate business objects from DLL metadata")
    parser.add_argument("--merge-strategy", choices=["merge", "replace"], default="merge",
                       help="Merge strategy: merge (default) or replace")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum confidence threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    generator = BusinessObjectsGenerator()
    generator.generate_and_merge(
        merge_strategy=args.merge_strategy,
        min_confidence=args.min_confidence
    )


if __name__ == "__main__":
    main()

