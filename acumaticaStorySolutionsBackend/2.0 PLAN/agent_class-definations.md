# **🔥 AGENT CLASS DEFINITIONS (PYTHON)**

Each agent follows a strict structure:

- `run()` — main execution
    
- Accepts structured input
    
- Returns structured output
    
- Uses injected tools (retriever, vector search, symbol index, semantic index, etc.)
    

Below is the full set.

---

# **0. Shared Data Models (recommended)**

`from dataclasses import dataclass, field from typing import List, Dict, Any, Optional  # --------------------- Shared Models ----------------------  @dataclass class IntentSpec:     story_type: str     business_concepts: List[str]     required_operations: List[str]     ac_items: List[str]     priority_signals: List[str]     risk_level: str  @dataclass class EntityMap:     primary_dac: str     primary_graph: str     primary_screen_id: str     line_dac: Optional[str] = None     custom_fields: Dict[str, str] = field(default_factory=dict)     key_fields: Dict[str, str] = field(default_factory=dict)  @dataclass class RetrievalPlan:     manuals: List[str]     dlls: List[str]     tags: List[str]     use_vision: bool  @dataclass class SolutionBlueprint:     pattern: str     handlers_required: List[str]     steps: List[str]  @dataclass class CandidateSolution:     code: str     explanation: str     config_steps: List[str]  @dataclass class ValidationResult:     success: bool     errors: List[str]     warnings: List[str]`

---

# **1. Story Interpreter Agent**

`class StoryInterpreterAgent:     def __init__(self, llm):         self.llm = llm      def run(self, story_text: str, ac_text: str) -> IntentSpec:         """         Extract story type, business concepts, AC items, required operations.         """         prompt = f"""         Extract structured intent from the following story.          STORY:         {story_text}          AC:         {ac_text}          Return JSON with:         - story_type         - business_concepts         - required_operations         - ac_items         - priority_signals         - risk_level         """         response = self.llm.prompt(prompt)         data = response.json()          return IntentSpec(**data)`

---

# **2. Domain Resolver Agent**

Uses:

- `semantic_index.json`
    
- symbol index built from DLL reflection
    
- name scoring / alias matching
    
- strict validation (no guessing)
    

`class DomainResolverAgent:     def __init__(self, semantic_index, symbol_index, llm):         self.semantic_index = semantic_index         self.symbol_index = symbol_index         self.llm = llm      def run(self, intent: IntentSpec) -> EntityMap:         """         Map natural language concepts -> actual DACs, Graphs, Fields.         """          hits = []         for obj in self.semantic_index["objects"]:             if any(alias.lower() in intent.business_concepts[0].lower()                    for alias in obj["aliases"] + [obj["concept"]]):                 hits.append(obj)          if not hits:             raise ValueError(f"No semantic index match for concept {intent.business_concepts}")          best = sorted(hits, key=lambda x: x.get("priority", 0), reverse=True)[0]          # Validate DAC exists in DLL symbol table         if best["primary_dac"] not in self.symbol_index["classes"]:             raise ValueError(f"DAC {best['primary_dac']} not found in symbol_index")          return EntityMap(             primary_dac=best["primary_dac"],             primary_graph=best["primary_graph"],             primary_screen_id=best["primary_screen_id"],             line_dac=best.get("line_dac"),             custom_fields=best.get("custom_fields", {}),             key_fields=best.get("key_fields", {})         )`

---

# **3. Retrieval Strategist Agent**

`class RetrievalStrategistAgent:     def __init__(self, domain_index, manual_index, dll_index):         self.domain_index = domain_index         self.manual_index = manual_index         self.dll_index = dll_index      def run(self, intent: IntentSpec, entity_map: EntityMap) -> RetrievalPlan:         """         Pick retrieval sources based on intent and entity grounding.         """          manuals = []         dlls = []         tags = []          # Logic: add manuals based on story type         if intent.story_type == "email_template":             manuals.append("EmailTemplates.pdf")             tags.append("email")          if "validation" in intent.required_operations:             manuals.append("CustomizationGuide.pdf")             dlls.append(entity_map.primary_dac.split('.')[0])          # Add module-aware sources         if entity_map.primary_dac in self.dll_index["custom_modules"]:             dlls.append(self.dll_index["custom_modules"][entity_map.primary_dac])          return RetrievalPlan(             manuals=list(set(manuals)),             dlls=list(set(dlls)),             tags=tags,             use_vision=True if not manuals else False         )`

---

# **4. Solution Architect Agent**

`class SolutionArchitectAgent:     def __init__(self, patterns, llm):         self.patterns = patterns         self.llm = llm      def run(self, intent: IntentSpec, entity_map: EntityMap) -> SolutionBlueprint:         """         Select the correct Acumatica development pattern.         """          pattern = self.patterns.detect(intent, entity_map)          steps = self.patterns.get_steps(pattern)          handlers = self.patterns.required_handlers(pattern)          return SolutionBlueprint(             pattern=pattern,             handlers_required=handlers,             steps=steps         )`

---

# **5. Code/Configuration Generator Agent**

`class CodeGeneratorAgent:     def __init__(self, templates, llm):         self.templates = templates         self.llm = llm      def run(self, blueprint: SolutionBlueprint, entity_map: EntityMap) -> CandidateSolution:         """         Fill deterministic templates with DACs, fields, graphs.         """          template = self.templates.get(blueprint.pattern)          filled_code = template.render(             dac=entity_map.primary_dac,             graph=entity_map.primary_graph,             line_dac=entity_map.line_dac,             custom_fields=entity_map.custom_fields         )          # Optional LLM polishing step         explanation = self.llm.prompt(f"Explain this solution:\n{filled_code}").text          return CandidateSolution(             code=filled_code,             explanation=explanation,             config_steps=[]         )`

---

# **6. Validator Agent**

`class ValidatorAgent:     def __init__(self, symbol_index, semantic_index, patterns):         self.symbol_index = symbol_index         self.semantic_index = semantic_index         self.patterns = patterns      def run(self, candidate: CandidateSolution, blueprint: SolutionBlueprint, entity_map: EntityMap) -> ValidationResult:         errors = []         warnings = []          # Validate DACs         if entity_map.primary_dac not in self.symbol_index["classes"]:             errors.append(f"DAC {entity_map.primary_dac} not in symbol index")          # Validate fields         for field in entity_map.custom_fields.values():             if field and field not in self.symbol_index["fields"]:                 warnings.append(f"Field {field} missing in symbol index")          # Validate pattern integrity         if not self.patterns.validate(blueprint.pattern, candidate.code):             errors.append("Generated code does not match pattern integrity")          return ValidationResult(             success=len(errors) == 0,             errors=errors,             warnings=warnings         )`

---

# **7. Orchestrator (Pipeline Controller)**

`class Orchestrator:     def __init__(self, agents):         self.story_interpreter = agents["interpreter"]         self.domain_resolver = agents["resolver"]         self.retrieval_strategist = agents["retriever"]         self.solution_architect = agents["architect"]         self.code_generator = agents["generator"]         self.validator = agents["validator"]      def run(self, story, ac):         # 1. Interpret         intent = self.story_interpreter.run(story, ac)          # 2. Resolve domain         entity_map = self.domain_resolver.run(intent)          # 3. Retrieval plan (optional: execute vector search here)         retrieval_plan = self.retrieval_strategist.run(intent, entity_map)          # 4. Architect solution         blueprint = self.solution_architect.run(intent, entity_map)          # 5. Generate code/config         candidate = self.code_generator.run(blueprint, entity_map)          # 6. Validate         validation = self.validator.run(candidate, blueprint, entity_map)          # 7. Final output         if not validation.success:             candidate.explanation += "\nVALIDATION FAILED:\n" + "\n".join(validation.errors)          return candidate, validation`

---

# **8. Required Tools You Must Provide**

### Your system must provide:

|Tool|Purpose|
|---|---|
|**LLM client**|Story extraction, explanation|
|**Semantic Object Index** (`business_objects.json`)|Concept → DAC/Graph mapping|
|**Symbol Index** (`symbol_index.json`)|Classes, fields from DLLs|
|**Pattern Library**|Auto-line, validation, email-template, etc.|
|**Template Library**|Deterministic code templates|
|**Vector Search Client**|PDF/DLL chunk retrieval|

---

# **9. Optional: Vector Search Integration Stub**

`class VectorSearchTool:     def search(self, query, tags=None, top_k=15):         return []  # your implementation here`

Agents can use this for deep retrieval.