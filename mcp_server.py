import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

# PDF processing
import PyPDF2
import fitz  # PyMuPDF for better text extraction

# Vector embeddings and similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP()

# Initialize components
class RAGSystem:
    def __init__(self, storage_dir: str = "rag_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage paths
        self.documents_dir = self.storage_dir / "documents"
        self.chunks_dir = self.storage_dir / "chunks"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.index_file = self.storage_dir / "faiss_index.bin"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # Create directories
        for dir_path in [self.documents_dir, self.chunks_dir, self.embeddings_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
        # Initialize FAISS index
        self.index = self._load_or_create_index()
    
    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"documents": {}, "chunks": {}, "next_chunk_id": 0}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _load_or_create_index(self):
        if self.index_file.exists():
            return faiss.read_index(str(self.index_file))
        else:
            # Create new index with dimension 384 (all-MiniLM-L6-v2 embedding size)
            return faiss.IndexFlatIP(384)
    
    def _save_index(self):
        faiss.write_index(self.index, str(self.index_file))

# Initialize RAG system
rag_system = RAGSystem()

#### Tools ####

@mcp.tool()
def upload_pdf(file_path: str, document_name: Optional[str] = None) -> Dict[str, Any]:
    """Upload and process a PDF file for RAG system"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if not file_path.suffix.lower() == '.pdf':
            return {"error": "File must be a PDF"}
        
        # Generate document ID
        doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_name = document_name or file_path.stem
        
        # Extract text from PDF
        text_content = _extract_pdf_text(file_path)
        if not text_content.strip():
            return {"error": "No text content found in PDF"}
        
        # Save original document
        doc_file = rag_system.documents_dir / f"{doc_id}.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Create chunks
        chunks = _create_text_chunks(text_content)
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = rag_system.metadata["next_chunk_id"]
            rag_system.metadata["next_chunk_id"] += 1
            
            # Save chunk
            chunk_file = rag_system.chunks_dir / f"{chunk_id}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Generate embedding
            embedding = rag_system.embedding_model.encode([chunk])[0]
            
            # Add to FAISS index
            rag_system.index.add(embedding.reshape(1, -1))
            
            # Save embedding
            np.save(rag_system.embeddings_dir / f"{chunk_id}.npy", embedding)
            
            # Update metadata
            rag_system.metadata["chunks"][str(chunk_id)] = {
                "document_id": doc_id,
                "chunk_index": i,
                "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "created_at": datetime.now().isoformat()
            }
            
            chunk_ids.append(chunk_id)
        
        # Update document metadata
        rag_system.metadata["documents"][doc_id] = {
            "name": doc_name,
            "original_path": str(file_path),
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat()
        }
        
        # Save metadata and index
        rag_system._save_metadata()
        rag_system._save_index()
        
        return {
            "success": True,
            "document_id": doc_id,
            "document_name": doc_name,
            "chunks_created": len(chunks),
            "message": f"Successfully processed PDF: {doc_name}"
        }
        
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}

@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for relevant document chunks using semantic similarity"""
    try:
        if rag_system.index.ntotal == 0:
            return {"error": "No documents in the system. Please upload some PDFs first."}
        
        # Generate query embedding
        query_embedding = rag_system.embedding_model.encode([query])[0]
        
        # Search in FAISS index
        scores, indices = rag_system.index.search(query_embedding.reshape(1, -1), min(top_k, rag_system.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            # Load chunk text
            chunk_file = rag_system.chunks_dir / f"{idx}.txt"
            if chunk_file.exists():
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
                
                # Get chunk metadata
                chunk_meta = rag_system.metadata["chunks"].get(str(idx), {})
                doc_meta = rag_system.metadata["documents"].get(chunk_meta.get("document_id", ""), {})
                
                results.append({
                    "chunk_id": idx,
                    "score": float(score),
                    "text": chunk_text,
                    "document_name": doc_meta.get("name", "Unknown"),
                    "document_id": chunk_meta.get("document_id"),
                    "chunk_index": chunk_meta.get("chunk_index")
                })
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

@mcp.tool()
def list_documents() -> Dict[str, Any]:
    """List all uploaded documents"""
    try:
        documents = []
        for doc_id, doc_info in rag_system.metadata["documents"].items():
            documents.append({
                "document_id": doc_id,
                "name": doc_info["name"],
                "chunk_count": doc_info["chunk_count"],
                "created_at": doc_info["created_at"]
            })
        
        return {
            "success": True,
            "documents": documents,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        return {"error": f"Failed to list documents: {str(e)}"}

@mcp.tool()
def delete_document(document_id: str) -> Dict[str, Any]:
    """Delete a document and its associated chunks"""
    try:
        if document_id not in rag_system.metadata["documents"]:
            return {"error": f"Document not found: {document_id}"}
        
        doc_info = rag_system.metadata["documents"][document_id]
        chunk_ids = doc_info["chunk_ids"]
        
        # Delete chunk files and embeddings
        for chunk_id in chunk_ids:
            chunk_file = rag_system.chunks_dir / f"{chunk_id}.txt"
            embedding_file = rag_system.embeddings_dir / f"{chunk_id}.npy"
            
            if chunk_file.exists():
                chunk_file.unlink()
            if embedding_file.exists():
                embedding_file.unlink()
            
            # Remove from metadata
            if str(chunk_id) in rag_system.metadata["chunks"]:
                del rag_system.metadata["chunks"][str(chunk_id)]
        
        # Delete document file
        doc_file = rag_system.documents_dir / f"{document_id}.txt"
        if doc_file.exists():
            doc_file.unlink()
        
        # Remove from metadata
        del rag_system.metadata["documents"][document_id]
        
        # Rebuild FAISS index (simple approach - could be optimized)
        rag_system.index = faiss.IndexFlatIP(384)
        for chunk_id in rag_system.metadata["chunks"].keys():
            embedding_file = rag_system.embeddings_dir / f"{chunk_id}.npy"
            if embedding_file.exists():
                embedding = np.load(embedding_file)
                rag_system.index.add(embedding.reshape(1, -1))
        
        # Save metadata and index
        rag_system._save_metadata()
        rag_system._save_index()
        
        return {
            "success": True,
            "message": f"Successfully deleted document: {doc_info['name']}"
        }
        
    except Exception as e:
        return {"error": f"Failed to delete document: {str(e)}"}

@mcp.tool()
def get_rag_stats() -> Dict[str, Any]:
    """Get statistics about the RAG system"""
    try:
        total_docs = len(rag_system.metadata["documents"])
        total_chunks = len(rag_system.metadata["chunks"])
        
        # Calculate storage usage
        storage_size = sum(f.stat().st_size for f in rag_system.storage_dir.rglob('*') if f.is_file())
        
        return {
            "success": True,
            "statistics": {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "storage_size_bytes": storage_size,
                "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                "embedding_dimension": 384,
                "storage_directory": str(rag_system.storage_dir)
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to get statistics: {str(e)}"}

# Helper functions
def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF using PyMuPDF for better quality"""
    try:
        doc = fitz.open(file_path)
        text_content = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_content += page.get_text()
            text_content += "\n\n"  # Add page separator
        
        doc.close()
        return text_content.strip()
        
    except Exception as e:
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                    text_content += "\n\n"
                
                return text_content.strip()
        except Exception as e2:
            raise Exception(f"Failed to extract text with both PyMuPDF and PyPDF2: {str(e2)}")

def _create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Create overlapping text chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:  # Only if we find a good break point
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk.strip()]

#### Resources ####

@mcp.resource("rag://documents")
def get_documents_resource() -> str:
    """Get list of all documents in the RAG system"""
    try:
        docs = []
        for doc_id, doc_info in rag_system.metadata["documents"].items():
            docs.append(f"- {doc_info['name']} (ID: {doc_id}, Chunks: {doc_info['chunk_count']})")
        
        if not docs:
            return "No documents uploaded yet."
        
        return "Documents in RAG system:\n" + "\n".join(docs)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

@mcp.resource("rag://document/{document_id}")
def get_document_content(document_id: str) -> str:
    """Get the full content of a specific document"""
    try:
        if document_id not in rag_system.metadata["documents"]:
            return f"Document not found: {document_id}"
        
        doc_file = rag_system.documents_dir / f"{document_id}.txt"
        if not doc_file.exists():
            return f"Document file not found: {document_id}"
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc_info = rag_system.metadata["documents"][document_id]
        header = f"Document: {doc_info['name']}\n"
        header += f"Created: {doc_info['created_at']}\n"
        header += f"Chunks: {doc_info['chunk_count']}\n"
        header += "=" * 50 + "\n\n"
        
        return header + content
    except Exception as e:
        return f"Error retrieving document: {str(e)}"

@mcp.resource("rag://stats")
def get_rag_stats_resource() -> str:
    """Get RAG system statistics"""
    try:
        stats = get_rag_stats()
        if "error" in stats:
            return stats["error"]
        
        s = stats["statistics"]
        return f"""RAG System Statistics:
- Total Documents: {s['total_documents']}
- Total Chunks: {s['total_chunks']}
- Storage Size: {s['storage_size_mb']} MB
- Embedding Dimension: {s['embedding_dimension']}
- Storage Directory: {s['storage_directory']}"""
    except Exception as e:
        return f"Error retrieving statistics: {str(e)}"

#### Prompts ####

@mcp.prompt()
def rag_query_prompt(query: str, context_chunks: str) -> List[tuple]:
    """Generate a prompt for RAG-based question answering"""
    return [
        ("system", "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer questions. If the context doesn't contain enough information to answer the question, say so clearly."),
        ("user", f"Context:\n{context_chunks}\n\nQuestion: {query}\n\nPlease answer the question based on the provided context.")
    ]

@mcp.prompt()
def document_summary_prompt(document_content: str) -> str:
    """Generate a prompt for document summarization"""
    return f"Please provide a comprehensive summary of the following document:\n\n{document_content}"

@mcp.prompt()
def search_suggestions_prompt(query: str, available_documents: str) -> List[tuple]:
    """Generate search suggestions based on available documents"""
    return [
        ("system", "You are a helpful assistant that suggests better search queries based on available documents."),
        ("user", f"Available documents:\n{available_documents}\n\nUser query: '{query}'\n\nSuggest 3-5 alternative or refined search queries that might yield better results from these documents.")
    ]


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='sse')