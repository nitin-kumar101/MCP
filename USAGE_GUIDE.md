# MCP RAG System - Usage Guide

## 🚀 Quick Start

### 1. Start the Server
```bash
python mcp_server.py
```
The server will start on `http://localhost:8000` and display:
- Loading of the SentenceTransformer model
- Initialization of the RAG storage system
- Server startup confirmation

### 2. Test the Connection
```bash
python test_server_simple.py
```
This will verify that all components are working correctly.

### 3. Use the Interactive Client
```bash
python interactive_client.py
```

## 📋 Available Commands

### `upload` - Upload PDF Files
```
rag> upload
Enter PDF file path: C:\path\to\your\document.pdf
Document name (default: document): My Research Paper
```

**What happens:**
- PDF text is extracted using PyMuPDF/PyPDF2
- Text is split into overlapping chunks (1000 chars, 200 overlap)
- Each chunk gets a vector embedding using SentenceTransformers
- Embeddings are stored in FAISS index for fast search
- Metadata is saved for document management

### `search` - Semantic Search
```
rag> search
Enter search query: machine learning applications
Number of results (default: 5): 3
```

**What happens:**
- Query is converted to vector embedding
- FAISS performs similarity search against all chunks
- Results are ranked by cosine similarity score
- Returns relevant text chunks with context

### `list` - View All Documents
```
rag> list
```
Shows all uploaded documents with:
- Document ID (for deletion/reference)
- Document name
- Number of chunks created
- Upload timestamp

### `stats` - System Statistics
```
rag> stats
```
Displays:
- Total documents and chunks
- Storage size in MB
- Embedding dimensions
- Storage directory location

### `delete` - Remove Documents
```
rag> delete
Enter document ID to delete: abc123...
Delete document abc123...? (y/N): y
```

**What happens:**
- Removes document and all associated chunks
- Deletes embedding vectors from FAISS index
- Cleans up storage files
- Updates metadata

## 🔧 Technical Details

### Storage Structure
```
rag_storage/
├── documents/          # Original extracted text files
├── chunks/            # Individual text chunks
├── embeddings/        # Numpy arrays of vector embeddings
├── faiss_index.bin    # FAISS vector search index
└── metadata.json      # Document and chunk metadata
```

### Embedding Model
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Language**: English (multilingual support available)
- **Performance**: Fast inference, good quality

### Text Processing
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Boundary Detection**: Sentence-aware splitting
- **Encoding**: UTF-8 with fallback handling

## 📄 PDF Processing

### Supported Features
- Text extraction from standard PDFs
- Multi-page documents
- Various PDF formats and encodings
- Automatic fallback between extraction methods

### Limitations
- Password-protected PDFs not supported
- Image-only PDFs require OCR (not included)
- Very large PDFs may require chunking

## 🔍 Search Capabilities

### Query Types
- **Keyword Search**: "machine learning"
- **Phrase Search**: "natural language processing"
- **Concept Search**: "AI applications in healthcare"
- **Question Format**: "What are the main findings?"

### Search Quality
- Semantic similarity (not just keyword matching)
- Context-aware results
- Ranked by relevance score (0.0 to 1.0)
- Handles synonyms and related concepts

## 🛠 Troubleshooting

### Server Won't Start
```bash
# Check if port 8000 is in use
netstat -an | findstr :8000

# Kill existing process if needed
taskkill /F /PID <process_id>
```

### Connection Errors
- Verify server is running: `python test_server_simple.py`
- Check firewall settings
- Ensure all dependencies are installed

### PDF Upload Fails
- Verify file path is correct
- Check file is not password-protected
- Ensure sufficient disk space
- Try different PDF if extraction fails

### Search Returns No Results
- Upload documents first with `upload` command
- Try broader search terms
- Check document content matches query language
- Verify embeddings were created successfully

### Memory Issues
- Reduce chunk size in server configuration
- Process fewer documents at once
- Monitor system memory usage
- Consider using GPU acceleration

## 🔧 Configuration

### Modify Chunk Size
Edit `mcp_server.py`:
```python
def _create_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
```

### Change Embedding Model
Edit `RAGSystem.__init__()`:
```python
self.embedding_model = SentenceTransformer('your-model-name')
```

### Custom Storage Location
```python
rag_system = RAGSystem(storage_dir="custom_storage_path")
```

## 📊 Performance Tips

### For Large Document Collections
1. Use GPU acceleration: `pip install faiss-gpu`
2. Batch process uploads
3. Consider IndexIVFFlat for faster search
4. Implement result caching

### For Better Search Quality
1. Use domain-specific embedding models
2. Adjust chunk size based on document type
3. Implement query expansion
4. Add metadata filtering

## 🔗 Integration Examples

### Use with LLM for RAG
```python
# 1. Search for relevant context
search_results = await session.call_tool("search_documents", {
    "query": user_question,
    "top_k": 5
})

# 2. Combine results into context
context = "\n\n".join([r["text"] for r in search_results["results"]])

# 3. Generate RAG prompt
prompt = await session.get_prompt("rag_query_prompt", {
    "query": user_question,
    "context_chunks": context
})

# 4. Send to LLM for final answer
```

### Batch Processing
```python
# Upload multiple PDFs
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for pdf_file in pdf_files:
    await session.call_tool("upload_pdf", {"file_path": pdf_file})
```

## 📝 Best Practices

### Document Management
- Use descriptive document names
- Organize by topic or date
- Regular cleanup of unused documents
- Monitor storage usage

### Search Optimization
- Start with broad queries, then narrow down
- Use multiple search terms
- Review search scores for quality assessment
- Combine results from multiple queries

### System Maintenance
- Regular backups of `rag_storage/` directory
- Monitor system performance
- Update embedding models as needed
- Clean up temporary files

## 🆘 Support

### Common Issues
1. **Unicode errors**: Use UTF-8 compatible terminal
2. **Path issues**: Use absolute paths for PDF files
3. **Permission errors**: Check file/directory permissions
4. **Memory errors**: Reduce batch sizes or chunk count

### Getting Help
- Check error messages in server logs
- Verify all dependencies are installed correctly
- Test with simple PDF files first
- Review configuration settings

This RAG system provides a solid foundation for document search and retrieval that can be extended and customized for specific use cases.
