# JIRA Story Solutions Service

A FastAPI service that processes JIRA stories and generates comprehensive markdown solutions using RAG (Retrieval-Augmented Generation) with HuggingFace embeddings and OpenAI.

## Overview

This service follows the workflow:
1. **JIRA Story Input** → JSON with description, acceptance criteria, and optional images
2. **Question Extraction** → LLM extracts/generates key questions from the story
3. **Knowledge Base Search** → RAG + Vision searches manuals for answers
4. **Answer Generation** → Accurate answers generated from knowledge base
5. **Solution Generation** → LLM creates coherent narrative solution
6. **Markdown Output** → Formatted solution document

## Architecture

```
acumaticaStorySolutions/
├── main.py                    # FastAPI application
├── requirements.txt           # Dependencies
├── ingest_cli.py              # CLI for ingesting PDF manuals
├── src/
│   ├── config/
│   │   └── config.py          # Configuration
│   ├── core/
│   │   ├── story_processor.py # JIRA story processing
│   │   ├── rag_service.py     # RAG wrapper (uses VDRInferencer)
│   │   └── solution_generator.py # Narrative solution generation
│   ├── api/
│   │   ├── models/
│   │   │   ├── requests.py    # Request models
│   │   │   └── responses.py   # Response models
│   │   └── routes/
│   │       └── solutions.py   # API routes
│   └── utils/
│       ├── logger_utils.py    # Logging utilities
│       ├── local_manager.py   # Knowledge base management
│       └── markdown_formatter.py # Markdown formatting
└── knowledge_base/
    └── manuals/               # Processed manuals (same structure as pdfRagAcumatica)
        └── [document_name]/
            ├── data/
            ├── images/
            ├── metadata/
            └── vectors/
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   API_KEY=your_api_key_for_authentication
   ```

3. **Ingest Manuals**
   ```bash
   python ingest_cli.py path/to/manual.pdf
   # Or process a directory:
   python ingest_cli.py path/to/manuals_directory/
   ```

4. **Run the Service**
   ```bash
   python main.py
   ```
   Service runs on `http://localhost:8001`

## API Endpoints

### POST `/solutions/process`
Process a JIRA story and generate solution.

**Request Body:**
```json
{
  "description": "As a user, I want to process sales returns...",
  "acceptance_criteria": [
    "User can create a return order",
    "System validates return eligibility"
  ],
  "images": [],
  "story_id": "STORY-123",
  "title": "Sales Return Processing"
}
```

**Response:**
```json
{
  "success": true,
  "story_id": "STORY-123",
  "solution_markdown": "# Solution for Story STORY-123\n\n...",
  "key_questions": ["How do I process returns?", ...],
  "question_answers": [...],
  "sources": [...],
  "processing_time": 12.5,
  "metadata": {...}
}
```

### GET `/solutions/health`
Health check endpoint.

### GET `/health`
General health check.

## Key Features

- **Question Extraction**: Uses LLM to identify key questions from JIRA stories
- **RAG Integration**: Uses VDRInferencer for knowledge base search with hybrid retrieval
- **Vision Analysis**: Analyzes document images for better understanding
- **Narrative Generation**: Creates coherent, structured solutions
- **Markdown Output**: Well-formatted solution documents

## Knowledge Base

The service uses a dedicated knowledge base at `knowledge_base/manuals/`.

To add manuals:
1. Place PDF files in `knowledge_base/manuals/pdfs/` directory (or specify path)
2. Run `python ingest_cli.py` (processes all PDFs in `knowledge_base/manuals/pdfs/`)
   Or: `python ingest_cli.py path/to/manual.pdf` (processes specific PDF)
3. Manuals are processed and stored in `knowledge_base/manuals/[document_name]/`

## Architecture Details

This is a **standalone, independent service** with:
- Self-contained VDR (Visual Document Retrieval) implementation
- Hybrid retrieval system combining multiple search strategies
- Local knowledge base management
- Complete RAG pipeline with vision analysis

## Notes

- Uses HuggingFace embeddings (free) for semantic search
- Uses OpenAI for LLM and Vision analysis
- Port 8001
- All processing is local (no cloud storage dependencies)
- Fully independent - no external project dependencies required

