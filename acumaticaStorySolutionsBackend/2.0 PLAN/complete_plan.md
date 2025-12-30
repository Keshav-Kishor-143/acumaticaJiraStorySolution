# **ACUMATICA SOLUTION GENERATOR 2.0 — FULL ARCHITECTURE PLAN**

## **“Agent-Orchestrated, Semantically-Grounded, Deterministic Acumatica Reasoning Engine”**

---

# **0. High-Level Vision**

Transform your current “single LLM + RAG” pipeline into a **multi-stage intelligence system** that mimics a senior Acumatica consultant’s thought process:

1. Understand the story
    
2. Identify the Acumatica objects
    
3. Retrieve only relevant info
    
4. Match correct patterns
    
5. Generate deterministic solutions
    
6. Validate technical correctness
    
7. Produce final answer
    

This architecture ensures **repeatable, correct, AC-aligned, module-specific Acumatica solutions**.

---

# **1. Core Components Overview**

Your new architecture has **four pillars**:

1. **Semantic Object Layer (“Understanding the Acumatica universe”)**
    
2. **Agent Orchestration Layer (“Thinking step-by-step like a consultant”)**
    
3. **Hybrid Retrieval Layer (“Pulling the exact content needed”)**
    
4. **Deterministic Pattern + Template Layer (“Generating reliable code/config”)**
    

And one **Quality Control Layer** for validation.

---

# **2. The Complete Architecture Diagram (Mental Model)**

`───────────────────────────────────────────────────────────────                 ACUMATICA SOLUTION GENERATOR 2.0 ───────────────────────────────────────────────────────────────                   ┌────────────────────────┐                  │  STORY INTERPRETER     │                  │ (Intent + AC parsing)  │                  └───────────┬────────────┘                              ▼                  ┌────────────────────────┐                  │   DOMAIN RESOLVER      │                  │ (Semantic Object Index │                  │   + symbol grounding)  │                  └───────────┬────────────┘                              ▼                  ┌────────────────────────┐                  │  RETRIEVAL STRATEGIST  │                  │ (KB prefilter + tags + │                  │   DLL/manual routing)  │                  └───────────┬────────────┘                              ▼                  ┌────────────────────────┐                  │   SOLUTION ARCHITECT   │                  │ (Pattern selection +   │                  │    Blueprint design)   │                  └───────────┬────────────┘                              ▼                  ┌────────────────────────┐                  │ CODE/CONFIG GENERATOR  │                  │ (Deterministic         │                  │   templates)           │                  └───────────┬────────────┘                              ▼                  ┌────────────────────────┐                  │       VALIDATOR        │                  │ (AC, entity, logic,    │                  │  template integrity)   │                  └───────────┬────────────┘                              ▼                      FINAL SOLUTION ───────────────────────────────────────────────────────────────`

---

# **3. Component-by-Component Detailed Architecture**

---

# **3.1 Story Interpreter Agent**

**Goal:** Convert raw user story → structured, machine-readable IntentSpec.

### Inputs

- Raw story text
    
- Acceptance criteria
    

### Outputs (**IntentSpec**)

`{   "story_type": "email_template" | "validation" | "workflow" | "auto_line_creation" | "calculation",   "business_concepts": ["Rental Return", "Damage Fee"],   "required_operations": ["line_insertion", "amount_calculation"],   "ac_items": [...],   "priority_signals": ["date", "billing_contact", "condition update"],   "risk_level": "low|medium|high" }`

### Responsibilities

- Extract business intent
    
- Detect story type (config vs code)
    
- Identify verbal hints
    
- Extract AC as structured data
    
- Send to Domain Resolver
    

This agent makes sure your system **interprets stories reliably**.

---

# **3.2 Domain Resolver Agent**

**Goal:** Ground the story in your installation’s **real DACs, Graphs, Screens, Fields**  
using the **Semantic Object Index**.

### Inputs

- IntentSpec
    
- business_objects.json (Semantic Object Index)
    
- DLL-based Symbol Index (class names, fields, etc.)
    

### Outputs (**EntityMap**)

`{   "primary_dac": "NVRTReturnTicket",   "primary_graph": "NVRTReturnTicketEntry",   "line_dac": "NVRTReturnTicketTran",   "key_fields": {      "customer": "CustomerID",      "date": "DocDate"   },   "custom_fields": {      "condition": "UsrEquipmentCondition"   } }`

### Responsibilities

- Match “Rental Return” → NVRTReturnTicket
    
- Confirm presence of fields via Symbol Index
    
- Reject missing entities
    
- Provide fully grounded metadata to next layer
    

This prevents “guess DACs/graphs” errors.

---

# **3.3 Retrieval Strategist Agent**

**Goal:** Select the **right documents** to retrieve _before_ semantic search happens.

### Inputs

- EntityMap
    
- IntentSpec
    

### Output (**RetrieverPlan**)

- Which manuals to search
    
- Which DLLs to search
    
- Which tags to boost
    
- Whether to use Vision fallback
    

### Rules

- If story_type = “email_template” → search only email manuals + CRM docs
    
- If story_type = “validation_rule” → search DACs + Customization Guide
    
- If DAC belongs to custom module NV.Rental360 → prioritize NV DLL folders
    
- If EntityMap has line_dac → include SO/FS logic
    

Your current retrieval is blind.  
This layer makes it **intent-driven** and **entity-driven**.

---

# **3.4 Solution Architect Agent**

**Goal:** Select the correct Acumatica pattern and produce a solution blueprint.

### Inputs

- IntentSpec
    
- EntityMap
    
- RetrieverPlan
    
- Pattern library
    

### Outputs (**SolutionBlueprint**)

`{   "pattern": "auto_line_creation",   "handlers_required": ["FieldUpdated", "RowPersisting"],   "blueprint_steps": [      "Detect condition update",      "Insert damage fee line if missing",      "Calculate fee amount",      "Enforce AC validations"   ] }`

### Responsibilities

- Classify the problem
    
- Pick one of ~70 patterns (you will build these)
    
- Produce a step-by-step plan for solution
    
- Validate that DAC/Graph supports required events
    

This is where your tool starts sounding like a senior consultant.

---

# **3.5 Code/Configuration Generator Agent**

**Goal:** Convert Blueprint → actual implementation using deterministic templates.

### Inputs

- SolutionBlueprint
    
- EntityMap
    
- Pattern library templates
    
- Field mappings
    

### Output

- C# code OR
    
- Email template OR
    
- Workflow steps OR
    
- Automation steps
    

### Templates

- Validation Rule Template
    
- Date Range Template
    
- Auto-Line Insert Template
    
- Calculation Template
    
- Email Template HTML Template
    
- Workflow Action Template
    

### Guarantees

- Correct class names
    
- Correct DAC fields
    
- Correct handlers
    
- No placeholders
    
- No hallucination
    

This solves your inconsistent code generation.

---

# **3.6 Validator Agent**

**Goal:** Validate the final solution **before** returning it.

### Validates:

- DAC exists in Symbol Index
    
- Graph exists
    
- Field names exist
    
- Event handlers make sense
    
- AC conditions satisfied
    
- Blueprint followed
    
- Template integrity
    
- No TODOs or placeholders
    

If it fails → send feedback back to **Solution Architect Agent**.

This is the missing step that ChatGPT implicitly does but your system doesn’t.

---

# **4. The Semantic Object Layer**

This is the **heart of the architecture**.

Without this file, your system will _always_ ask:

“What DAC does Rental Return refer to?”  
“What Graph does EX Order use?”

Because Acumatica’s object model is not universal.

### Files included:

- `business_objects.json` (Semantic Object Index)
    
- `symbol_index.json` (parsed DLL metadata)
    
- `form_index.json` (PDF → Form name mappings)
    
- `module_index.json` (module → DLL relationships)
    

These files anchor your pipeline to **real classes**.

---

# **5. Hybrid Retrieval Layer**

Includes:

- Vector search
    
- Metadata filtering
    
- Semantic tag boosting
    
- Module-aware scoring
    
- DLL-aware scoring
    
- Vision fallback for diagrams/pages without text
    

Goal: pull exactly what an Acumatica consultant would reference.

---

# **6. Deterministic Pattern + Template Layer**

Contains reusable patterns for:

- RowPersisting validation
    
- FieldUpdated recalculation
    
- Auto-line creation
    
- Date-range validation
    
- Amount computations
    
- Email template mapping
    
- Workflow actions
    
- Approval rules
    
- Business events
    
- ARM report logic
    
- GI conditions
    
- Custom selector logic
    

This replaces “guessing” with “controlled generation.”

---

# **7. End-to-End Flow (Master Diagram)**

User Story
    ▼
Story Interpreter
    ▼
Domain Resolver ───────────────────────┐
    ▼                                  │ uses
Retrieval Strategist                   │ Semantic Index
    ▼                                  │
Retriever → KB → Chunks                │
    ▼                                  │
Solution Architect                     │
    ▼                                  │ ensures
Generator → Deterministic Templates    │ Entity Correctness
    ▼                                  │
Validator ────────────────recursive────┘
    ▼
Final Solution (Accurate, AC-driven, Reliable)


# **8. What You Gain**

### ✔ Zero hallucination

### ✔ Zero misidentified DAC/Graph

### ✔ No unnecessary clarifications

### ✔ High accuracy even for custom modules

### ✔ Enterprise-grade predictable output

### ✔ Blueprint-level reasoning before code

### ✔ Complete alignment with Acceptance Criteria

This architecture puts your system at **ChatGPT-level reasoning** with **Acumatica-specific determinism**.