#!/usr/bin/env python3
"""
DLL Processor for Acumatica Knowledge Base Integration

Extracts structured information from .NET DLL assemblies:
- Class definitions, namespaces, inheritance hierarchies
- Methods, properties, attributes
- Generates visual diagrams for Vision analysis
- Converts code structure to searchable text and images
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import io
import textwrap

# Try to import pythonnet for .NET interop
try:
    import clr
    PYTHONNET_AVAILABLE = True
except ImportError:
    PYTHONNET_AVAILABLE = False

from src.utils.logger_utils import get_logger
from src.config.config import config

logger = get_logger("DLL_PROCESSOR")


class DLLProcessor:
    """
    Extract structured information from Acumatica DLL files
    for knowledge base ingestion with Vision support
    """
    
    def __init__(self):
        self.logger = logger
        self.pythonnet_available = PYTHONNET_AVAILABLE
        
        if not self.pythonnet_available:
            self.logger.warning(
                "pythonnet not available. DLL processing will use fallback methods.",
                extra={"fallback": True}
            )
    
    def process_dll(self, dll_path: str) -> Dict[str, Any]:
        """
        Extract all useful information from a DLL file
        
        Returns structured data that can be converted to text/images
        for embedding generation and Vision analysis
        """
        dll_path = Path(dll_path)
        if not dll_path.exists():
            raise FileNotFoundError(f"DLL not found: {dll_path}")
        
        self.logger.info("Processing DLL file", extra={
            "dll_path": str(dll_path),
            "dll_name": dll_path.name
        })
        
        extracted_data = {
            "dll_name": dll_path.stem,
            "dll_path": str(dll_path),
            "file_size": dll_path.stat().st_size,
            "assembly_name": None,
            "version": None,
            "namespaces": {},
            "classes": [],
            "methods": [],
            "properties": [],
            "attributes": [],
            "inheritance_hierarchies": {},
            "processing_method": "reflection" if self.pythonnet_available else "metadata_only"
        }
        
        if self.pythonnet_available:
            try:
                extracted_data = self._extract_via_reflection(dll_path, extracted_data)
            except Exception as e:
                self.logger.error("Reflection extraction failed, using fallback", extra={
                    "error": str(e),
                    "dll_path": str(dll_path)
                })
                extracted_data = self._extract_via_metadata(dll_path, extracted_data)
        else:
            extracted_data = self._extract_via_metadata(dll_path, extracted_data)
        
        self.logger.info("DLL processing completed", extra={
            "dll_name": extracted_data["dll_name"],
            "namespaces_count": len(extracted_data["namespaces"]),
            "classes_count": len(extracted_data["classes"]),
            "methods_count": len(extracted_data["methods"])
        })
        
        return extracted_data
    
    def _load_assembly(self, dll_path: Path):
        """Load .NET assembly using multiple fallback methods"""
        import System  # type: ignore
        from System.Reflection import Assembly  # type: ignore
        import os
        import tempfile
        import shutil
        
        dll_path_str = str(dll_path.absolute())
        
        # Method 1: Try loading from byte array (bypasses all file system security)
        try:
            with open(dll_path_str, 'rb') as f:
                dll_bytes = f.read()
            net_bytes = System.Array[System.Byte](dll_bytes)
            assembly = Assembly.Load(net_bytes)
            self.logger.debug("Successfully loaded DLL using byte array method")
            return assembly
        except Exception as byte_load_error:
            # Method 2: Copy to temp directory and load from there
            self.logger.debug("Byte array load failed, trying temp directory method", extra={"error": str(byte_load_error)})
            temp_dir = tempfile.mkdtemp(prefix="dll_processing_")
            temp_dll = os.path.join(temp_dir, dll_path.name)
            
            try:
                shutil.copy2(dll_path_str, temp_dll)
                try:
                    assembly = Assembly.LoadFile(temp_dll)
                except Exception as load_file_error:
                    self.logger.debug("LoadFile failed, trying LoadFrom", extra={"error": str(load_file_error)})
                    assembly = Assembly.LoadFrom(temp_dll)
                return assembly
            finally:
                try:
                    if os.path.exists(temp_dll):
                        os.remove(temp_dll)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    self.logger.warning("Failed to cleanup temp files", extra={"error": str(cleanup_error)})
    
    def _load_dependencies(self, dll_path: Path):
        """Load common Acumatica DLL dependencies"""
        import System  # type: ignore
        from System.Reflection import Assembly  # type: ignore
        
        dll_dir = dll_path.parent
        dependency_order = ["PX.Common.dll", "PX.Common.Std.dll", "PX.Data.dll", "PX.Data.BQL.Fluent.dll"]
        
        for dep_name in dependency_order:
            dep_path = dll_dir / dep_name
            if dep_path.exists() and dep_path != dll_path:
                try:
                    with open(str(dep_path.absolute()), 'rb') as f:
                        dep_bytes = f.read()
                    dep_net_bytes = System.Array[System.Byte](dep_bytes)
                    dep_assembly = Assembly.Load(dep_net_bytes)
                    self.logger.debug(f"Loaded dependency: {dep_name}", extra={"assembly": dep_assembly.GetName().Name})
                except Exception as dep_error:
                    self.logger.debug(f"Could not load dependency {dep_name}", extra={"error": str(dep_error)})
                    continue
    
    def _get_assembly_types(self, assembly):
        """Extract all types from assembly, handling ReflectionTypeLoadException"""
        try:
            return assembly.GetTypes()
        except System.Reflection.ReflectionTypeLoadException as type_load_ex:  # type: ignore
            types = [t for t in type_load_ex.Types if t is not None]
            loader_errors = type_load_ex.LoaderExceptions
            self.logger.warning("Some types failed to load (likely missing dependencies)", extra={
                "loaded_types": len(types),
                "failed_types": len([e for e in loader_errors if e is not None]),
                "assembly": assembly.GetName().Name
            })
            return types
    
    def _process_methods_list(self, methods, type_obj, extracted_data: Dict[str, Any]) -> tuple[int, int]:
        """Process a list of methods and extract them. Returns (extracted_count, skipped_count)"""
        methods_extracted = 0
        methods_skipped_special = 0
        
        for method in methods:
            if method.IsSpecialName:
                methods_skipped_special += 1
                continue
            
            try:
                method_info = self._extract_method_info(method, type_obj)
                extracted_data["methods"].append(method_info)
                methods_extracted += 1
            except Exception as extract_error:
                self.logger.debug(f"Failed to extract method {method.Name} from {type_obj.Name}", 
                                extra={"error": str(extract_error)})
                continue
        
        return methods_extracted, methods_skipped_special
    
    def _extract_methods_with_binding_flags(self, type_obj, extracted_data: Dict[str, Any]) -> tuple[bool, Optional[Exception]]:
        """Extract methods using BindingFlags. Returns (success, error)."""
        try:
            from System.Reflection import BindingFlags  # type: ignore
            all_methods = type_obj.GetMethods(
                BindingFlags.Public | 
                BindingFlags.NonPublic | 
                BindingFlags.Static | 
                BindingFlags.Instance |
                BindingFlags.DeclaredOnly
            )
            
            methods_extracted, methods_skipped_special = self._process_methods_list(
                all_methods, type_obj, extracted_data
            )
            
            if methods_extracted > 0 or methods_skipped_special > 0:
                self.logger.debug(f"Extracted {methods_extracted} methods from {type_obj.Name} "
                                f"(skipped {methods_skipped_special} special names)")
            return True, None
        except Exception as method_error:
            return False, method_error
    
    def _extract_methods_fallback(self, type_obj, extracted_data: Dict[str, Any], method_error: Optional[Exception]):
        """Fallback method extraction using default GetMethods()"""
        try:
            error_msg = str(method_error) if method_error else "Unknown error"
            self.logger.debug(f"BindingFlags approach failed for {type_obj.Name}, trying default GetMethods()", 
                            extra={"error": error_msg})
            methods = type_obj.GetMethods()
            self._process_methods_list(methods, type_obj, extracted_data)
        except Exception as fallback_error:
            self.logger.debug(f"Failed to extract methods from {type_obj.Name}", 
                            extra={"error": str(fallback_error)})
    
    def _extract_type_methods(self, type_obj, extracted_data: Dict[str, Any]):
        """Extract methods from a type object"""
        success, error = self._extract_methods_with_binding_flags(type_obj, extracted_data)
        if not success:
            self._extract_methods_fallback(type_obj, extracted_data, error)
    
    def _extract_type_properties(self, type_obj, extracted_data: Dict[str, Any]):
        """Extract properties from a type object"""
        try:
            for prop in type_obj.GetProperties():
                prop_info = self._extract_property_info(prop, type_obj)
                extracted_data["properties"].append(prop_info)
        except Exception as prop_error:
            self.logger.debug(f"Failed to extract properties from {type_obj.Name}", extra={"error": str(prop_error)})
    
    def _process_type(self, type_obj, extracted_data: Dict[str, Any]):
        """Process a single type: extract class info, methods, and properties"""
        if type_obj is None:
            return
        
        try:
            namespace = type_obj.Namespace or "Global"
            class_info = self._extract_class_info(type_obj)
            
            if namespace not in extracted_data["namespaces"]:
                extracted_data["namespaces"][namespace] = []
            
            extracted_data["namespaces"][namespace].append(class_info)
            extracted_data["classes"].append(class_info)
            
            # Track inheritance
            if type_obj.BaseType:
                base_name = type_obj.BaseType.FullName
                if base_name not in extracted_data["inheritance_hierarchies"]:
                    extracted_data["inheritance_hierarchies"][base_name] = []
                extracted_data["inheritance_hierarchies"][base_name].append(type_obj.FullName)
            
            # Extract methods and properties
            self._extract_type_methods(type_obj, extracted_data)
            self._extract_type_properties(type_obj, extracted_data)
            
        except Exception as e:
            self.logger.debug(f"Failed to process type: {type_obj.Name}", extra={"error": str(e)})
    
    def _extract_via_reflection(self, dll_path: Path, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DLL information using .NET reflection via pythonnet"""
        try:
            assembly = self._load_assembly(dll_path)
            
            extracted_data["assembly_name"] = assembly.GetName().Name
            extracted_data["version"] = str(assembly.GetName().Version)
            
            self._load_dependencies(dll_path)
            types = self._get_assembly_types(assembly)
            
            for type_obj in types:
                self._process_type(type_obj, extracted_data)
                    
        except Exception as e:
            self.logger.error("Reflection extraction error", extra={"error": str(e)})
            raise
        
        return extracted_data
    
    def _extract_via_metadata(self, dll_path: Path, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: Extract basic metadata without reflection"""
        extracted_data["assembly_name"] = dll_path.stem
        extracted_data["version"] = "Unknown"
        extracted_data["processing_method"] = "metadata_only"
        
        # Try to extract namespace hints from filename
        # Acumatica DLLs follow patterns like PX.Objects.dll, PX.Data.dll
        if "." in dll_path.stem:
            parts = dll_path.stem.split(".")
            if len(parts) >= 2:
                namespace_hint = ".".join(parts[:-1])
                extracted_data["namespaces"][namespace_hint] = []
                self.logger.info("Extracted namespace hint from filename", extra={
                    "namespace": namespace_hint
                })
        
        return extracted_data
    
    def _extract_class_info(self, type_obj) -> Dict[str, Any]:
        """Extract class-level information"""
        return {
            "name": type_obj.Name,
            "full_name": type_obj.FullName,
            "namespace": type_obj.Namespace,
            "base_type": type_obj.BaseType.FullName if type_obj.BaseType else None,
            "is_abstract": type_obj.IsAbstract,
            "is_public": type_obj.IsPublic,
            "is_class": type_obj.IsClass,
            "is_interface": type_obj.IsInterface,
            "interfaces": [i.FullName for i in type_obj.GetInterfaces()],
            "custom_attributes": self._extract_attributes(type_obj.GetCustomAttributes(False))
        }
    
    def _extract_method_info(self, method, parent_type) -> Dict[str, Any]:
        """Extract method information"""
        params = []
        try:
            for p in method.GetParameters():
                params.append({
                    "name": p.Name,
                    "type": str(p.ParameterType),
                    "is_optional": p.IsOptional
                })
        except Exception as param_error:
            self.logger.debug(f"Failed to extract parameters from method {method.Name}", extra={"error": str(param_error)})
        
        # Calculate IsOverride: a method is an override if it's virtual and doesn't have NewSlot flag
        is_override = False
        try:
            from System.Reflection import MethodAttributes  # type: ignore
            attrs = method.Attributes
            # IsOverride = IsVirtual AND NOT NewSlot
            is_override = (attrs & MethodAttributes.Virtual) != 0 and (attrs & MethodAttributes.NewSlot) == 0
        except Exception:
            # Fallback: if virtual and not abstract, likely an override
            try:
                is_override = method.IsVirtual and not method.IsAbstract
            except Exception:
                pass
        
        return {
            "name": method.Name,
            "parent_class": parent_type.FullName,
            "return_type": str(method.ReturnType),
            "parameters": params,
            "is_public": method.IsPublic,
            "is_virtual": method.IsVirtual,
            "is_override": is_override,
            "is_static": method.IsStatic,
            "custom_attributes": self._extract_attributes(method.GetCustomAttributes(False))
        }
    
    def _extract_property_info(self, prop, parent_type) -> Dict[str, Any]:
        """Extract property information"""
        return {
            "name": prop.Name,
            "parent_class": parent_type.FullName,
            "property_type": str(prop.PropertyType),
            "can_read": prop.CanRead,
            "can_write": prop.CanWrite,
            "custom_attributes": self._extract_attributes(prop.GetCustomAttributes(False))
        }
    
    def _extract_attributes(self, attributes) -> List[str]:
        """Extract attribute names"""
        return [str(attr.GetType().Name) for attr in attributes]
    
    def _format_assembly_info(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Format assembly header information"""
        return [
            f"# Assembly: {extracted_data['assembly_name']}",
            f"Version: {extracted_data.get('version', 'Unknown')}",
            f"DLL: {extracted_data['dll_name']}",
            ""
        ]
    
    def _format_class_info(self, class_info: Dict[str, Any]) -> List[str]:
        """Format individual class information"""
        lines = []
        class_type = "Interface" if class_info.get('is_interface') else "Class"
        lines.append(f"**{class_type}: {class_info['name']}**")
        lines.append(f"- Full Name: `{class_info['full_name']}`")
        
        if class_info.get('base_type'):
            lines.append(f"- Inherits from: `{class_info['base_type']}`")
        
        if class_info.get('interfaces'):
            interfaces_str = ', '.join([f'`{i}`' for i in class_info['interfaces'][:5]])
            lines.append(f"- Implements: {interfaces_str}")
        
        if class_info.get('custom_attributes'):
            attrs_str = ', '.join(class_info['custom_attributes'][:10])
            lines.append(f"- Attributes: {attrs_str}")
        
        lines.append("")
        return lines
    
    def _format_namespaces_and_classes(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Format namespaces and classes section"""
        lines = ["## Namespaces and Classes", ""]
        
        for namespace, classes in sorted(extracted_data["namespaces"].items()):
            lines.append(f"### Namespace: {namespace}")
            lines.append("")
            
            for class_info in classes[:50]:  # Limit per namespace
                lines.extend(self._format_class_info(class_info))
        
        return lines
    
    def _format_methods(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Format methods section"""
        lines = ["## Key Methods", ""]
        
        for method in extracted_data["methods"][:100]:
            params_str = ", ".join([f"{p['type']} {p['name']}" for p in method['parameters'][:3]])
            if len(method['parameters']) > 3:
                params_str += "..."
            
            modifiers = []
            if method.get('is_public'):
                modifiers.append("public")
            if method.get('is_virtual'):
                modifiers.append("virtual")
            if method.get('is_static'):
                modifiers.append("static")
            
            modifier_str = " ".join(modifiers) + " " if modifiers else ""
            method_line = f"- `{modifier_str}{method['parent_class']}.{method['name']}({params_str})` → `{method['return_type']}`"
            lines.append(method_line)
        
        return lines
    
    def _format_properties(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Format properties section"""
        lines = ["## Key Properties", ""]
        
        for prop in extracted_data["properties"][:100]:
            if prop['can_read'] and prop['can_write']:
                access = "read-write"
            elif prop['can_read']:
                access = "read-only"
            else:
                access = "write-only"
            
            prop_line = f"- `{prop['parent_class']}.{prop['name']}`: `{prop['property_type']}` ({access})"
            lines.append(prop_line)
        
        return lines
    
    def _format_inheritance_hierarchies(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Format inheritance hierarchies section"""
        lines = []
        
        if extracted_data.get("inheritance_hierarchies"):
            lines.extend(["## Inheritance Hierarchies", ""])
            for base_class, derived_classes in list(extracted_data["inheritance_hierarchies"].items())[:20]:
                lines.append(f"**Base: `{base_class}`**")
                for derived in derived_classes[:10]:
                    lines.append(f"  - `{derived}`")
                lines.append("")
        
        return lines
    
    def convert_to_text(self, extracted_data: Dict[str, Any]) -> str:
        """
        Convert extracted DLL data to searchable text format
        This text will be embedded and indexed
        """
        text_parts = []
        text_parts.extend(self._format_assembly_info(extracted_data))
        text_parts.extend(self._format_namespaces_and_classes(extracted_data))
        text_parts.extend(self._format_methods(extracted_data))
        text_parts.extend(self._format_properties(extracted_data))
        text_parts.extend(self._format_inheritance_hierarchies(extracted_data))
        
        return "\n".join(text_parts)
    
    def _load_fonts(self, font_name: str = "arial.ttf") -> tuple:
        """Load fonts with fallback to default. Returns (font_large, font_medium, font_small)."""
        try:
            font_large = ImageFont.truetype(font_name, 24)
            font_medium = ImageFont.truetype(font_name, 18)
            font_small = ImageFont.truetype(font_name, 14)
        except OSError:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        return font_large, font_medium, font_small
    
    def _group_classes_by_namespace(self, classes: List[Dict[str, Any]], max_classes: int) -> Dict[str, List[Dict[str, Any]]]:
        """Group classes by namespace."""
        namespace_groups = {}
        for class_info in classes[:max_classes]:
            namespace = class_info.get('namespace', 'Global')
            if namespace not in namespace_groups:
                namespace_groups[namespace] = []
            namespace_groups[namespace].append(class_info)
        return namespace_groups
    
    def _draw_namespace_header(self, draw, namespace: str, x_margin: int, y_position: int, 
                               img_width: int, font_medium) -> int:
        """Draw namespace header and return new y_position."""
        draw.rectangle(
            [x_margin, y_position, img_width - x_margin, y_position + 40],
            fill='#E0E0E0',
            outline='black'
        )
        draw.text((x_margin + 10, y_position + 10), f"Namespace: {namespace}", 
                 fill='black', font=font_medium)
        return y_position + 50
    
    def _draw_class_box(self, draw, class_info: Dict[str, Any], x_class: int, y_position: int,
                       box_width: int, box_height: int, font_medium, font_small):
        """Draw a single class box with name and base type."""
        class_name = class_info['name']
        class_type = "I" if class_info.get('is_interface') else "C"
        
        # Draw class box
        draw.rectangle(
            [x_class, y_position, x_class + box_width, y_position + box_height],
            fill='#F5F5F5',
            outline='#333333',
            width=2
        )
        
        # Class name with type indicator
        draw.text(
            (x_class + 10, y_position + 10),
            f"[{class_type}] {class_name}",
            fill='black',
            font=font_medium
        )
        
        # Base type if exists
        if class_info.get('base_type'):
            base_name = class_info['base_type'].split('.')[-1]
            draw.text(
                (x_class + 10, y_position + 40),
                f"extends {base_name}",
                fill='#666666',
                font=font_small
            )
    
    def _draw_namespace_classes(self, draw, classes: List[Dict[str, Any]], x_margin: int,
                                y_position: int, img_width: int, img_height: int,
                                font_medium, font_small) -> int:
        """Draw classes in a namespace and return final y_position."""
        box_width = 300
        box_height = 80
        x_class = x_margin + 20
        
        for i, class_info in enumerate(classes[:10]):
            if y_position > img_height - 200:
                break
            
            if x_class + box_width > img_width - x_margin:
                x_class = x_margin + 20
                y_position += box_height + 20
            
            self._draw_class_box(draw, class_info, x_class, y_position, 
                               box_width, box_height, font_medium, font_small)
            
            x_class += box_width + 20
            
            if i % 3 == 2:  # New row every 3 classes
                x_class = x_margin + 20
                y_position += box_height + 20
        
        return y_position + box_height + 40
    
    def generate_class_diagram_image(self, extracted_data: Dict[str, Any], output_path: Path, max_classes: int = 30) -> Optional[Path]:
        """
        Generate a visual class diagram image for Vision analysis
        Creates a hierarchical diagram showing class relationships
        """
        try:
            # Create image with sufficient size
            img_width = 2400
            img_height = 3200
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            font_large, font_medium, font_small = self._load_fonts()
            
            y_position = 50
            x_margin = 50
            
            # Title
            title = f"Class Diagram: {extracted_data['assembly_name']}"
            draw.text((x_margin, y_position), title, fill='black', font=font_large)
            y_position += 60
            
            # Group classes by namespace
            namespace_groups = self._group_classes_by_namespace(
                extracted_data["classes"], max_classes
            )
            
            # Draw namespace sections
            for namespace, classes in list(namespace_groups.items())[:5]:
                y_position = self._draw_namespace_header(
                    draw, namespace, x_margin, y_position, img_width, font_medium
                )
                y_position = self._draw_namespace_classes(
                    draw, classes, x_margin, y_position, img_width, img_height,
                    font_medium, font_small
                )
            
            # Save image
            img.save(output_path, 'PNG', quality=95)
            self.logger.info("Generated class diagram image", extra={
                "output_path": str(output_path),
                "classes_shown": min(max_classes, len(extracted_data["classes"]))
            })
            
            return output_path
            
        except Exception as e:
            self.logger.error("Failed to generate class diagram", extra={
                "error": str(e),
                "output_path": str(output_path)
            })
            return None
    
    def generate_code_structure_image(self, extracted_data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Generate a formatted code structure image showing key classes and methods
        Vision can analyze this to understand code organization
        """
        try:
            img_width = 2400
            img_height = 3200
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                font_code = ImageFont.truetype("consola.ttf", 16) or ImageFont.truetype("courier.ttf", 16)
            except OSError:
                font_code = ImageFont.load_default()
            
            y_position = 50
            x_margin = 50
            line_height = 25
            
            # Title
            title = f"Code Structure: {extracted_data['assembly_name']}"
            draw.text((x_margin, y_position), title, fill='black', font=font_code)
            y_position += 50
            
            # Show top classes with their key methods
            for class_info in extracted_data["classes"][:20]:
                if y_position > img_height - 100:
                    break
                
                # Class header
                class_name = class_info['full_name']
                draw.text((x_margin, y_position), f"class {class_name} {{", fill='#0066CC', font=font_code)
                y_position += line_height
                
                # Show key methods for this class
                class_methods = [m for m in extracted_data["methods"] 
                                if m['parent_class'] == class_info['full_name']][:5]
                
                for method in class_methods:
                    params = ", ".join([f"{p['type'].split('.')[-1]} {p['name']}" 
                                       for p in method['parameters'][:3]])
                    method_line = f"  {method['return_type'].split('.')[-1]} {method['name']}({params});"
                    # Wrap long lines
                    wrapped = textwrap.wrap(method_line, width=100)
                    for line in wrapped:
                        draw.text((x_margin + 20, y_position), line, fill='#333333', font=font_code)
                        y_position += line_height
                
                draw.text((x_margin, y_position), "}", fill='#0066CC', font=font_code)
                y_position += line_height + 10
            
            img.save(output_path, 'PNG', quality=95)
            self.logger.info("Generated code structure image", extra={
                "output_path": str(output_path)
            })
            
            return output_path
            
        except Exception as e:
            self.logger.error("Failed to generate code structure image", extra={
                "error": str(e)
            })
            return None

