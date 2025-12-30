#!/usr/bin/env python3
"""
Acumatica Patterns - Static map of common Acumatica DACs, Graphs, Forms, and aliases

Provides:
- Known DAC → Form → Graph relationships
- Common synonyms and aliases
- Alias resolution helpers
"""

from typing import Dict, List, Optional, Tuple
import re

from src.utils.logger_utils import get_logger


class AcumaticaPatterns:
    """Static patterns and mappings for Acumatica entities"""
    
    def __init__(self):
        self.logger = get_logger("ACUMATICA_PATTERNS")
        
        # DAC → Form → Graph mappings
        self.dac_form_graph_map = {
            # Sales Order
            'SOOrder': {
                'forms': ['SO301000'],
                'graphs': ['SOOrderEntry'],
                'aliases': ['Sales Order', 'SalesOrder', 'SO Order']
            },
            # Customer
            'Customer': {
                'forms': ['AR303000', 'CR303000'],
                'graphs': ['CustomerMaint'],
                'aliases': ['Customer', 'ARCustomer', 'CRCustomer']
            },
            # Event Order
            'EVOrder': {
                'forms': ['EV301000'],
                'graphs': ['EVOrderEntry'],
                'aliases': ['Event Order', 'EventOrder', 'EV Order']
            },
            # AR Invoice
            'ARInvoice': {
                'forms': ['AR301000'],
                'graphs': ['ARInvoiceEntry'],
                'aliases': ['Invoice', 'AR Invoice', 'Accounts Receivable Invoice']
            },
            # Purchase Order
            'POOrder': {
                'forms': ['PO301000'],
                'graphs': ['POOrderEntry'],
                'aliases': ['Purchase Order', 'PurchaseOrder', 'PO Order']
            },
            # Quote (CR Quote/Opportunity)
            'CRQuote': {
                'forms': ['CR306015'],
                'graphs': ['QuoteMaint'],
                'aliases': ['Quote', 'Sales Quote', 'CR Quote', 'Opportunity Quote', 'Email Quote']
            },
            # Quote (legacy - also map to CRQuote)
            'Quote': {
                'forms': ['CR306015', 'SO301000'],
                'graphs': ['QuoteMaint'],
                'aliases': ['Quote', 'Sales Quote', 'CR Quote', 'Opportunity Quote', 'Email Quote']
            },
            # Employee
            'EPEmployee': {
                'forms': ['EP203000'],
                'graphs': ['EmployeeMaint'],
                'aliases': ['Employee', 'EP Employee']
            },
            # Project
            'PMProject': {
                'forms': ['PM301000'],
                'graphs': ['ProjectEntry'],
                'aliases': ['Project', 'PM Project']
            },
            # Task
            'PMTask': {
                'forms': ['PM301000'],
                'graphs': ['TaskEntry'],
                'aliases': ['Task', 'PM Task']
            },
            # Time Entry
            'EPTimeEntry': {
                'forms': ['EP301000'],
                'graphs': ['TimeEntryMaint'],
                'aliases': ['Time Entry', 'TimeEntry', 'EP Time Entry']
            },
            # Expense Claim
            'EPExpenseClaim': {
                'forms': ['EP301000'],
                'graphs': ['ExpenseClaimEntry'],
                'aliases': ['Expense Claim', 'ExpenseClaim', 'EP Expense Claim']
            },
            # Inventory Item
            'InventoryItem': {
                'forms': ['IN202500'],
                'graphs': ['InventoryItemMaint'],
                'aliases': ['Item', 'Inventory Item', 'Stock Item']
            },
            # Stock Item
            'StockItem': {
                'forms': ['IN202500'],
                'graphs': ['StockItemMaint'],
                'aliases': ['Stock Item', 'StockItem']
            },
            # Shipment
            'SOShipment': {
                'forms': ['SO302000'],
                'graphs': ['ShipmentEntry'],
                'aliases': ['Shipment', 'SO Shipment', 'Sales Shipment']
            },
            # Payment
            'ARPayment': {
                'forms': ['AR302000'],
                'graphs': ['PaymentEntry'],
                'aliases': ['Payment', 'AR Payment', 'Customer Payment']
            }
        }
        
        # Form ID patterns
        self.form_patterns = {
            'SO': r'SO\d{6}',  # Sales Order forms
            'AR': r'AR\d{6}',  # Accounts Receivable forms
            'PO': r'PO\d{6}',  # Purchase Order forms
            'IN': r'IN\d{6}',  # Inventory forms
            'EP': r'EP\d{6}',  # Employee forms
            'PM': r'PM\d{6}',  # Project Management forms
            'EV': r'EV\d{6}',  # Event forms
            'CR': r'CR\d{6}',  # Customer forms
        }
        
        # Common field patterns
        self.field_patterns = {
            'order_number': ['OrderNbr', 'OrderNumber', 'Order Number'],
            'customer_id': ['CustomerID', 'Customer', 'CustID'],
            'order_date': ['OrderDate', 'Date', 'Order Date'],
            'status': ['Status', 'OrderStatus', 'Order Status'],
            'amount': ['Amount', 'OrderTotal', 'Total', 'Order Amount'],
            'description': ['Description', 'Descr', 'Order Description']
        }
        
        # Graph naming patterns
        self.graph_patterns = {
            'entry': r'\w+Entry',
            'maint': r'\w+Maint',
            'setup': r'\w+Setup',
            'inquiry': r'\w+Inquiry'
        }
        
        self.logger.info("Acumatica Patterns initialized", extra={
            "dac_count": len(self.dac_form_graph_map),
            "form_patterns": len(self.form_patterns)
        })
    
    def resolve_dac_alias(self, alias: str) -> Optional[str]:
        """
        Resolve a DAC alias to canonical DAC name
        
        Args:
            alias: Alias or synonym for DAC
            
        Returns:
            Canonical DAC name or None if not found
        """
        alias_lower = alias.lower().strip()
        
        for dac_name, dac_info in self.dac_form_graph_map.items():
            if alias_lower == dac_name.lower():
                return dac_name
            
            # Check aliases
            for dac_alias in dac_info['aliases']:
                if alias_lower == dac_alias.lower():
                    return dac_name
        
        return None
    
    def get_forms_for_dac(self, dac_name: str) -> List[str]:
        """
        Get form IDs for a DAC
        
        Args:
            dac_name: DAC name (can be alias)
            
        Returns:
            List of form IDs
        """
        canonical_dac = self.resolve_dac_alias(dac_name)
        if canonical_dac and canonical_dac in self.dac_form_graph_map:
            return self.dac_form_graph_map[canonical_dac]['forms']
        return []
    
    def get_graphs_for_dac(self, dac_name: str) -> List[str]:
        """
        Get graph names for a DAC
        
        Args:
            dac_name: DAC name (can be alias)
            
        Returns:
            List of graph names
        """
        canonical_dac = self.resolve_dac_alias(dac_name)
        if canonical_dac and canonical_dac in self.dac_form_graph_map:
            return self.dac_form_graph_map[canonical_dac]['graphs']
        return []
    
    def resolve_form_id(self, form_id: str) -> Optional[str]:
        """
        Normalize and validate form ID
        
        Args:
            form_id: Form ID (may have variations)
            
        Returns:
            Normalized form ID or None if invalid pattern
        """
        form_id = form_id.strip().upper()
        
        # Check if matches any form pattern
        for prefix, pattern in self.form_patterns.items():
            if re.match(pattern, form_id):
                return form_id
        
        return None
    
    def resolve_field_alias(self, field_alias: str) -> Optional[str]:
        """
        Resolve field alias to common field name
        
        Args:
            field_alias: Field alias or synonym
            
        Returns:
            Common field name or None
        """
        alias_lower = field_alias.lower().strip()
        
        for common_name, aliases in self.field_patterns.items():
            if alias_lower == common_name.lower():
                return aliases[0]  # Return first canonical name
            
            for alias in aliases:
                if alias_lower == alias.lower():
                    return aliases[0]  # Return first canonical name
        
        return None
    
    def get_all_dacs(self) -> List[str]:
        """Get list of all known DAC names"""
        return list(self.dac_form_graph_map.keys())
    
    def get_all_forms(self) -> List[str]:
        """Get list of all known form IDs"""
        forms = []
        for dac_info in self.dac_form_graph_map.values():
            forms.extend(dac_info['forms'])
        return sorted(set(forms))
    
    def get_all_graphs(self) -> List[str]:
        """Get list of all known graph names"""
        graphs = []
        for dac_info in self.dac_form_graph_map.values():
            graphs.extend(dac_info['graphs'])
        return sorted(set(graphs))
    
    def expand_entity(self, entity: str, entity_type: str) -> List[str]:
        """
        Expand an entity using patterns and aliases
        
        Args:
            entity: Entity name to expand
            entity_type: Type of entity ('dac', 'form', 'graph', 'field')
            
        Returns:
            List of expanded entity names (including original)
        """
        expanded = [entity]
        
        if entity_type == 'dac':
            canonical = self.resolve_dac_alias(entity)
            if canonical:
                expanded.append(canonical)
                # Add forms and graphs
                expanded.extend(self.get_forms_for_dac(canonical))
                expanded.extend(self.get_graphs_for_dac(canonical))
        
        elif entity_type == 'form':
            normalized = self.resolve_form_id(entity)
            if normalized:
                expanded.append(normalized)
        
        elif entity_type == 'field':
            canonical = self.resolve_field_alias(entity)
            if canonical:
                expanded.append(canonical)
        
        return list(set(expanded))  # Remove duplicates

