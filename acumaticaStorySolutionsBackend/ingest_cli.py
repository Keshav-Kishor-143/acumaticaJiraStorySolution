import sys
import json
import shutil
from pathlib import Path
import numpy as np
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.absolute()
sys.path.append(str(backend_dir))

# Add src to path
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

# Import from local modules
from src.core.ingest import VDRIngestor
from src.core.dll_processor import DLLProcessor
from src.config.config import config
from src.utils.logger_utils import get_logger

logger = get_logger("INGEST_CLI")


def check_directory_complete(base_dir: Path) -> bool:
    """Check if a directory has all required files and correct structure"""
    data_dir = base_dir / "data"
    images_dir = base_dir / "images"
    metadata_dir = base_dir / "metadata"
    vectors_dir = base_dir / "vectors"
    
    # Check if all directories exist
    if not all(d.exists() for d in [data_dir, images_dir, metadata_dir, vectors_dir]):
        return False
        
    # Check if metadata.json exists and has correct structure
    metadata_file = metadata_dir / "metadata.json"
    if not metadata_file.exists():
        return False
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
            if not metadata:  # Empty metadata
                return False
            # Check first entry for required fields
            first_entry = metadata[0]
            required_fields = ["page_number", "image_path", "pdf_name", "chunks"]
            if not all(field in first_entry for field in required_fields):
                return False
            # Check chunks structure
            chunks = first_entry["chunks"]
            if not chunks:  # Must have at least one chunk
                return False
            first_chunk = chunks[0]
            required_chunk_fields = ["vector_index", "coordinates"]
            if not all(field in first_chunk for field in required_chunk_fields):
                return False
            # Check coordinates structure
            coords = first_chunk["coordinates"]
            if not all(key in coords for key in ["x1", "y1", "x2", "y2"]):
                return False
    except:
        return False
        
    # Check if vectors.json exists and has embeddings
    vectors_file = vectors_dir / "vectors.json"
    if not vectors_file.exists():
        return False
    try:
        with open(vectors_file) as f:
            vectors = json.load(f)
            if not vectors or "embeddings" not in vectors:
                return False
            # Check that number of vectors matches total number of chunks across all pages
            total_chunks = sum(len(page["chunks"]) for page in metadata)
            if len(vectors["embeddings"]) != total_chunks:
                return False
    except:
        return False
        
    # Check if images exist and match metadata
    images = list(images_dir.glob("page*.jpg"))
    if not images:
        return False
    if len(images) != len(metadata):
        return False
        
    # Check if PDF exists in data directory
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        return False
        
    # All checks passed
    return True


def process_pdf(pdf_path: str, force: bool = False):
    """Process a PDF file and generate all necessary files"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False
        
    # Get PDF name without extension
    pdf_name = pdf_path.stem
    
    # Setup base directory in knowledge_base/manuals
    base_dir = backend_dir / "knowledge_base" / "manuals" / pdf_name
    
    # Check if already processed
    if not force and base_dir.exists() and check_directory_complete(base_dir):
        print(f"\n⏭️ Skipping {pdf_name} - already fully processed")
        print("To reprocess, use --force flag")
        return True
    
    # Setup directory structure
    data_dir = base_dir / "data"
    images_dir = base_dir / "images"
    metadata_dir = base_dir / "metadata"
    vectors_dir = base_dir / "vectors"
    
    # Create directories
    for dir_path in [data_dir, images_dir, metadata_dir, vectors_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting processing for PDF: {pdf_path.name}")
    print(f"📁 Directory structure created at: {base_dir}")
    
    # Copy PDF to data directory
    target_path = data_dir / pdf_path.name
    print(f"\n📋 Copying PDF to data directory: {target_path}")
    
    # If file exists and is the same, skip copying
    if target_path.exists():
        if target_path.stat().st_size == pdf_path.stat().st_size:
            print("✅ PDF already exists in data directory")
        else:
            try:
                shutil.copy2(pdf_path, target_path)
                print("✅ PDF copied to data directory")
            except PermissionError:
                print("⚠️ PDF already exists and is in use, proceeding with existing file")
    else:
        shutil.copy2(pdf_path, target_path)
        print("✅ PDF copied to data directory")
    
    # Initialize ingestor with environment variables
    os.environ["STORAGE_MODE"] = "local"
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["IMAGES_DIR"] = str(images_dir)
    os.environ["DPI"] = "200"
    os.environ["MAX_IMAGE_SIZE"] = "768"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["EMBEDDING_DIMENSION"] = "384"
    os.environ["TOP_K_RESULTS"] = "3"
    os.environ["USE_PRECOMPUTED_VECTORS"] = "true"
    os.environ["INCLUDE_VISION_ANALYSIS"] = "true"
    os.environ["VECTOR_DB_TYPE"] = "local"
    
    # Initialize ingestor
    ingestor = VDRIngestor()
    ingestor.images_dir = images_dir  # Override the default images directory
    
    # Check for existing images
    existing_images = list(images_dir.glob("page*.jpg"))
    if existing_images and not force:
        print(f"✅ Using {len(existing_images)} existing images")
        image_metadata = []
        for img_path in sorted(existing_images):
            page_num = int(img_path.stem.replace("page", ""))
            metadata = {
                "pdf_name": pdf_name,
                "pdf_path": str(target_path),
                "page_number": page_num,
                "image_path": str(img_path),
                "image_filename": img_path.name,
                "image_size": "unknown",
                "processing_mode": "fast_local_processing"
            }
            image_metadata.append(metadata)
    else:
        if force and existing_images:
            print("\n🗑️ Removing existing images...")
            for img in existing_images:
                img.unlink()
            print("✅ Existing images removed")
            
        print("\n🖼️ Converting PDF to images...")
        image_metadata = ingestor.convert_pdf_to_images(str(target_path))
        if not image_metadata:
            print("❌ Failed to convert PDF to images")
            return False
        print(f"✅ Generated {len(image_metadata)} images")
    
    # Generate embeddings
    print("\n🧠 Generating embeddings...")
    embeddings_data = ingestor.generate_visual_embeddings(image_metadata)
    if not embeddings_data:
        print("❌ Failed to generate embeddings")
        return False
    print(f"✅ Generated embeddings for {len(embeddings_data)} images")
    
    # Prepare metadata and vectors
    metadata = []
    vectors = {"embeddings": []}
    
    # Sort embeddings by page number
    sorted_embeddings = sorted(embeddings_data, key=lambda x: x["metadata"]["page_number"])
    
    vector_index = 0  # Track vector indices
    
    for item in sorted_embeddings:
        page_num = item["metadata"]["page_number"]
        
        # Get actual image dimensions from the saved image
        image_path = images_dir / f"page{page_num}.jpg"
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Get embeddings for this page
        page_embedding = item["embedding"]
        if hasattr(page_embedding, 'tolist'):
            page_embedding = page_embedding.tolist()
        
        # Split page into chunks (e.g., 4 quadrants)
        chunk_width = img_width // 2
        chunk_height = img_height // 2
        chunks = []
        
        # Create 4 chunks per page with their coordinates
        for i in range(2):  # rows
            for j in range(2):  # columns
                x1 = j * chunk_width
                y1 = i * chunk_height
                x2 = x1 + chunk_width
                y2 = y1 + chunk_height
                
                # Create chunk embedding (for now, use same embedding)
                vectors["embeddings"].append(page_embedding)
                
                chunks.append({
                    "vector_index": vector_index,
                    "coordinates": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })
                vector_index += 1
        
        # Add metadata entry for this page
        metadata.append({
            "page_number": page_num,
            "section_id": f"page_{page_num}",
            "image_path": f"images/page{page_num}.jpg",
            "pdf_name": pdf_name,
            "chunks": chunks,  # Multiple chunks per page
            "section_type": "page",
            "timestamp": "",
            "id": f"{pdf_name}_page_{page_num}"
        })
        
        # Verify alignment
        if len(vectors["embeddings"]) != vector_index:
            raise ValueError(f"Misalignment detected! Vector count {len(vectors['embeddings'])} doesn't match expected index {vector_index} for page {page_num}")
    
    # Save metadata and vectors
    metadata_file = metadata_dir / "metadata.json"
    vectors_file = vectors_dir / "vectors.json"
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_file}")
    
    with open(vectors_file, "w") as f:
        json.dump(vectors, f, indent=2)
    print(f"✅ Vectors saved to: {vectors_file}")
    
    print("\n✨ Processing completed successfully!")
    print(f"📂 Results stored in: {base_dir}\n")
    return True


def process_directory(pdf_dir: str, force: bool = False):
    """Process all PDFs in a directory"""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return
        
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in: {pdf_dir}")
        return
        
    print(f"\n📚 Found {len(pdfs)} PDFs to process:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")
    
    print("\n🔄 Starting batch processing...")
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Processing: {pdf.name}")
        print("=" * 80)
        success = process_pdf(str(pdf), force)
        if not success:
            print(f"❌ Failed to process: {pdf.name}")
        print("=" * 80)
    
    print("\n✨ Batch processing completed!")


def process_dll(dll_path: str, force: bool = False):
    """Process a DLL file and generate knowledge base content with Vision support"""
    dll_path = Path(dll_path)
    if not dll_path.exists():
        print(f"❌ DLL file not found: {dll_path}")
        return False
    
    if dll_path.suffix.lower() != '.dll':
        print(f"❌ Not a DLL file: {dll_path}")
        return False
    
    # Get DLL name without extension
    dll_name = dll_path.stem
    
    # Setup base directory in knowledge_base/manuals
    base_dir = backend_dir / "knowledge_base" / "manuals" / f"{dll_name}_DLL"
    
    # Check if already processed
    if not force and base_dir.exists() and (base_dir / "metadata" / "metadata.json").exists():
        print(f"\n⏭️ Skipping {dll_name} - already processed")
        print("To reprocess, use --force flag")
        return True
    
    # Setup directory structure (same as PDFs for compatibility)
    data_dir = base_dir / "data"
    images_dir = base_dir / "images"
    metadata_dir = base_dir / "metadata"
    vectors_dir = base_dir / "vectors"
    
    # Create directories
    for dir_path in [data_dir, images_dir, metadata_dir, vectors_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting DLL processing: {dll_path.name}")
    print(f"📁 Directory structure created at: {base_dir}")
    
    # Copy DLL to data directory
    target_path = data_dir / dll_path.name
    if not target_path.exists() or force:
        try:
            shutil.copy2(dll_path, target_path)
            print(f"✅ DLL copied to data directory: {target_path}")
        except Exception as e:
            print(f"⚠️ Failed to copy DLL: {e}")
            return False
    
    # Initialize DLL processor
    print("\n🔍 Extracting DLL structure...")
    dll_processor = DLLProcessor()
    
    try:
        extracted_data = dll_processor.process_dll(str(dll_path))
        print(f"✅ Extracted {len(extracted_data['classes'])} classes from {len(extracted_data['namespaces'])} namespaces")
    except Exception as e:
        print(f"❌ Failed to extract DLL structure: {e}")
        logger.error("DLL extraction failed", extra={"error": str(e), "dll_path": str(dll_path)})
        return False
    
    # Save extracted data as JSON
    extraction_file = metadata_dir / "dll_extraction.json"
    with open(extraction_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, default=str)
    print(f"✅ Saved extraction data to: {extraction_file}")
    
    # Convert to text for embedding
    print("\n📝 Converting to text format...")
    text_content = dll_processor.convert_to_text(extracted_data)
    text_file = data_dir / "dll_content.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text_content)
    print(f"✅ Saved text content to: {text_file}")
    
    # Generate visual diagrams for Vision analysis
    print("\n🎨 Generating visual diagrams for Vision analysis...")
    class_diagram_path = images_dir / "class_diagram.png"
    code_structure_path = images_dir / "code_structure.png"
    
    diagram_generated = False
    if dll_processor.generate_class_diagram_image(extracted_data, class_diagram_path):
        print(f"✅ Generated class diagram: {class_diagram_path}")
        diagram_generated = True
    
    if dll_processor.generate_code_structure_image(extracted_data, code_structure_path):
        print(f"✅ Generated code structure image: {code_structure_path}")
        diagram_generated = True
    
    if not diagram_generated:
        print("⚠️ Could not generate visual diagrams, proceeding with text-only")
    
    # Initialize ingestor for embeddings
    os.environ["STORAGE_MODE"] = "local"
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["IMAGES_DIR"] = str(images_dir)
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["EMBEDDING_DIMENSION"] = "384"
    
    ingestor = VDRIngestor()
    ingestor.images_dir = images_dir
    
    # Prepare image metadata for Vision analysis
    image_metadata = []
    
    # Add class diagram if generated
    if class_diagram_path.exists():
        image_metadata.append({
            "dll_name": dll_name,
            "pdf_name": dll_name,  # Add pdf_name for compatibility with embedding generation
            "dll_path": str(target_path),
            "page_number": 1,
            "image_path": str(class_diagram_path),
            "image_filename": "class_diagram.png",
            "image_type": "class_diagram",
            "image_size": "unknown",
            "processing_mode": "dll_visual_analysis"
        })
    
    # Add code structure if generated
    if code_structure_path.exists():
        image_metadata.append({
            "dll_name": dll_name,
            "pdf_name": dll_name,  # Add pdf_name for compatibility with embedding generation
            "dll_path": str(target_path),
            "page_number": 2,
            "image_path": str(code_structure_path),
            "image_filename": "code_structure.png",
            "image_type": "code_structure",
            "image_size": "unknown",
            "processing_mode": "dll_visual_analysis"
        })
    
    # Generate embeddings for images (Vision-ready)
    metadata = []
    vectors = {"embeddings": []}
    
    if image_metadata:
        print("\n🧠 Generating embeddings for visual diagrams...")
        embeddings_data = ingestor.generate_visual_embeddings(image_metadata)
        
        if embeddings_data:
            print(f"✅ Generated embeddings for {len(embeddings_data)} images")
            
            vector_index = 0
            for item in embeddings_data:
                page_num = item["metadata"]["page_number"]
                image_path = Path(item["metadata"]["image_path"])
                
                # Get image dimensions
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                except:
                    img_width, img_height = 2400, 3200
                
                # Get embedding
                page_embedding = item["embedding"]
                if hasattr(page_embedding, 'tolist'):
                    page_embedding = page_embedding.tolist()
                
                # Create chunks (similar to PDF processing)
                chunk_width = img_width // 2
                chunk_height = img_height // 2
                chunks = []
                
                for i in range(2):
                    for j in range(2):
                        x1 = j * chunk_width
                        y1 = i * chunk_height
                        x2 = x1 + chunk_width
                        y2 = y1 + chunk_height
                        
                        vectors["embeddings"].append(page_embedding)
                        
                        chunks.append({
                            "vector_index": vector_index,
                            "coordinates": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            }
                        })
                        vector_index += 1
                
                # Add metadata entry
                metadata.append({
                    "page_number": page_num,
                    "section_id": f"diagram_{page_num}",
                    "image_path": f"images/{item['metadata']['image_filename']}",
                    "pdf_name": dll_name,  # Use pdf_name field for compatibility
                    "dll_name": dll_name,
                    "image_type": item["metadata"].get("image_type", "diagram"),
                    "chunks": chunks,
                    "section_type": "dll_diagram",
                    "timestamp": "",
                    "id": f"{dll_name}_diagram_{page_num}"
                })
    
    # Also generate embeddings for text content (split into chunks)
    print("\n🧠 Generating embeddings for text content...")
    text_chunks = text_content.split("\n\n")  # Split by paragraphs
    text_chunks = [chunk for chunk in text_chunks if chunk.strip()]
    
    for i, chunk in enumerate(text_chunks[:50]):  # Limit to 50 chunks
        # Generate embedding for text chunk
        embedding = ingestor.embedding_model.get_text_embedding(chunk)
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        vectors["embeddings"].append(embedding)
        
        metadata.append({
            "page_number": len(image_metadata) + i + 1,
            "section_id": f"text_chunk_{i}",
            "image_path": None,
            "pdf_name": dll_name,
            "dll_name": dll_name,
            "text_content": chunk[:500],  # Store preview
            "chunks": [{
                "vector_index": len(vectors["embeddings"]) - 1,
                "coordinates": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
            }],
            "section_type": "dll_text",
            "timestamp": "",
            "id": f"{dll_name}_text_{i}"
        })
    
    # Save metadata and vectors
    metadata_file = metadata_dir / "metadata.json"
    vectors_file = vectors_dir / "vectors.json"
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Metadata saved to: {metadata_file}")
    
    with open(vectors_file, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2)
    print(f"✅ Vectors saved to: {vectors_file}")
    
    print("\n✨ DLL processing completed successfully!")
    print(f"📂 Results stored in: {base_dir}\n")
    print(f"📊 Summary:")
    print(f"   - Classes extracted: {len(extracted_data['classes'])}")
    print(f"   - Methods extracted: {len(extracted_data['methods'])}")
    print(f"   - Properties extracted: {len(extracted_data['properties'])}")
    print(f"   - Visual diagrams: {len([m for m in metadata if m.get('image_type')])}")
    print(f"   - Text chunks: {len([m for m in metadata if m.get('section_type') == 'dll_text'])}")
    print(f"   - Total embeddings: {len(vectors['embeddings'])}\n")
    
    return True


def process_directory(pdf_dir: str, force: bool = False):
    """Process all PDFs and DLLs in a directory"""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return
    
    pdfs = list(pdf_dir.glob("*.pdf"))
    dlls = list(pdf_dir.glob("*.dll"))
    
    if not pdfs and not dlls:
        print(f"❌ No PDFs or DLLs found in: {pdf_dir}")
        return
    
    print(f"\n📚 Found {len(pdfs)} PDFs and {len(dlls)} DLLs to process:")
    for pdf in pdfs:
        print(f"  - {pdf.name} (PDF)")
    for dll in dlls:
        print(f"  - {dll.name} (DLL)")
    
    print("\n🔄 Starting batch processing...")
    
    # Process PDFs
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Processing PDF: {pdf.name}")
        print("=" * 80)
        success = process_pdf(str(pdf), force)
        if not success:
            print(f"❌ Failed to process: {pdf.name}")
        print("=" * 80)
    
    # Process DLLs
    for i, dll in enumerate(dlls, 1):
        print(f"\n[{i}/{len(dlls)}] Processing DLL: {dll.name}")
        print("=" * 80)
        success = process_dll(str(dll), force)
        if not success:
            print(f"❌ Failed to process: {dll.name}")
        print("=" * 80)
    
    print("\n✨ Batch processing completed!")


def main():
    # Check for --force flag
    force = False
    args = sys.argv[1:]
    if "--force" in args:
        force = True
        args.remove("--force")
    
    if not args:
        # Default to processing all PDFs/DLLs in knowledge_base/manuals/pdfs (if exists)
        pdfs_dir = backend_dir / "knowledge_base" / "manuals" / "pdfs"
        if pdfs_dir.exists():
            print(f"No path provided, processing all files in: {pdfs_dir}")
            process_directory(pdfs_dir, force)
        else:
            print("No path provided and default directory not found.")
            print("Usage: python ingest_cli.py [path_to_pdf_or_dll_or_directory] [--force]")
            print("Supports: .pdf files, .dll files, or directories containing them")
    else:
        # Process single file or directory
        path = args[0]
        path_obj = Path(path)
        
        if path_obj.is_dir():
            process_directory(path, force)
        elif path_obj.suffix.lower() == '.dll':
            process_dll(path, force)
        elif path_obj.suffix.lower() == '.pdf':
            process_pdf(path, force)
        else:
            print(f"❌ Unsupported file type: {path_obj.suffix}")
            print("Supported types: .pdf, .dll")


if __name__ == "__main__":
    main()

