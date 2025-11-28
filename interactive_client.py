import asyncio
import json
import sys
from pathlib import Path
from mcp import ClientSession
from mcp.client.sse import sse_client


class RAGInteractiveClient:
    def __init__(self):
        self.server_url = "http://localhost:8000/sse"
    
    async def run(self):
        """Run the interactive RAG client"""
        print("RAG Interactive Client")
        print("Commands: upload, search, list, stats, help, quit")
        print("Type 'help' for detailed command information")
        
        try:
            async with sse_client(url=self.server_url) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    print("Connected to RAG server!")
                    
                    while True:
                        try:
                            command = input("\nrag> ").strip().lower()
                            
                            if command == "quit" or command == "exit":
                                print("Goodbye!")
                                break
                            
                            elif command == "help":
                                self.show_help()
                            
                            elif command == "upload":
                                await self.handle_upload(session)
                            
                            elif command == "search":
                                await self.handle_search(session)
                            
                            elif command == "list":
                                await self.handle_list(session)
                            
                            elif command == "stats":
                                await self.handle_stats(session)
                            
                            elif command == "delete":
                                await self.handle_delete(session)
                            
                            elif command == "":
                                continue
                            
                            else:
                                print(f"Unknown command: {command}")
                                print("Type 'help' for available commands")
                        
                        except KeyboardInterrupt:
                            print("\nUse 'quit' to exit")
                        except EOFError:
                            break
                        except Exception as e:
                            print(f"Error: {e}")
        
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            print("Make sure the server is running with: python mcp_server.py")
    
    def show_help(self):
        """Show help information"""
        print("\nAvailable Commands:")
        print("  upload  - Upload a PDF file to the RAG system")
        print("  search  - Search for documents using a query")
        print("  list    - List all uploaded documents")
        print("  stats   - Show system statistics")
        print("  delete  - Delete a document by ID")
        print("  help    - Show this help message")
        print("  quit    - Exit the client")
    
    async def handle_upload(self, session):
        """Handle PDF upload"""
        file_path = input("Enter PDF file path: ").strip().strip('"')
        
        if not file_path:
            print("No file path provided")
            return
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"File not found: {file_path}")
            return
        
        if not path_obj.suffix.lower() == '.pdf':
            print("File must be a PDF")
            return
        
        doc_name = input(f"Document name (default: {path_obj.stem}): ").strip()
        if not doc_name:
            doc_name = path_obj.stem
        
        try:
            print("Uploading and processing PDF...")
            result = await session.call_tool("upload_pdf", arguments={
                "file_path": file_path,
                "document_name": doc_name
            })
            
            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                print(f"Success: {response['message']}")
                print(f"Document ID: {response['document_id']}")
                print(f"Chunks created: {response['chunks_created']}")
        
        except Exception as e:
            print(f"Upload failed: {e}")
    
    async def handle_search(self, session):
        """Handle document search"""
        query = input("Enter search query: ").strip()
        
        if not query:
            print("No query provided")
            return
        
        try:
            top_k_input = input("Number of results (default: 5): ").strip()
            top_k = int(top_k_input) if top_k_input else 5
        except ValueError:
            top_k = 5
        
        try:
            print(f"Searching for: '{query}'...")
            result = await session.call_tool("search_documents", arguments={
                "query": query,
                "top_k": top_k
            })
            
            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                print(f"Found {response['total_results']} results:")
                for i, res in enumerate(response['results'], 1):
                    print(f"\n{i}. Score: {res['score']:.3f}")
                    print(f"   Document: {res['document_name']}")
                    print(f"   Chunk {res['chunk_index']}")
                    print(f"   Text: {res['text'][:200]}...")
        
        except Exception as e:
            print(f"Search failed: {e}")
    
    async def handle_list(self, session):
        """Handle document listing"""
        try:
            result = await session.call_tool("list_documents")
            response = json.loads(result.content[0].text)
            
            if "error" in response:
                print(f"Error: {response['error']}")
            elif response['total_documents'] == 0:
                print("No documents uploaded yet")
            else:
                print(f"Documents ({response['total_documents']}):")
                for doc in response['documents']:
                    print(f"  ID: {doc['document_id']}")
                    print(f"  Name: {doc['name']}")
                    print(f"  Chunks: {doc['chunk_count']}")
                    print(f"  Created: {doc['created_at']}")
                    print()
        
        except Exception as e:
            print(f"List failed: {e}")
    
    async def handle_stats(self, session):
        """Handle statistics display"""
        try:
            result = await session.call_tool("get_rag_stats")
            response = json.loads(result.content[0].text)
            
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                s = response['statistics']
                print("System Statistics:")
                print(f"  Total Documents: {s['total_documents']}")
                print(f"  Total Chunks: {s['total_chunks']}")
                print(f"  Storage Size: {s['storage_size_mb']} MB")
                print(f"  Embedding Dimension: {s['embedding_dimension']}")
                print(f"  Storage Directory: {s['storage_directory']}")
        
        except Exception as e:
            print(f"Stats failed: {e}")
    
    async def handle_delete(self, session):
        """Handle document deletion"""
        doc_id = input("Enter document ID to delete: ").strip()
        
        if not doc_id:
            print("No document ID provided")
            return
        
        confirm = input(f"Delete document {doc_id}? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Deletion cancelled")
            return
        
        try:
            result = await session.call_tool("delete_document", arguments={
                "document_id": doc_id
            })
            
            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                print(f"Success: {response['message']}")
        
        except Exception as e:
            print(f"Delete failed: {e}")


if __name__ == "__main__":
    client = RAGInteractiveClient()
    asyncio.run(client.run())
