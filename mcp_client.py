import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from mcp.client.sse import sse_client


class RAGClient:
    def __init__(self, server_url: str = "http://localhost:8000/sse"):
        self.server_url = server_url
    
    async def run_demo(self):
        """Run a comprehensive demo of the RAG system"""
        async with sse_client(url=self.server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                print("🚀 RAG System Demo Starting...")
                print("=" * 50)
                
                # 1. List available tools
                await self._list_tools(session)
                
                # 2. Check initial system stats
                await self._check_system_stats(session)
                
                # 3. List documents (should be empty initially)
                await self._list_documents(session)
                
                # 4. Demo PDF upload (you'll need to provide a PDF path)
                await self._demo_pdf_upload(session)
                
                # 5. Demo document search
                await self._demo_search(session)
                
                # 6. Demo resources
                await self._demo_resources(session)
                
                # 7. Demo prompts
                await self._demo_prompts(session)
                
                print("\n🎉 RAG System Demo Complete!")
    
    async def _list_tools(self, session):
        """List all available tools"""
        print("\n📋 Available Tools:")
        tools = await session.list_tools()
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
    
    async def _check_system_stats(self, session):
        """Check RAG system statistics"""
        print("\n📊 System Statistics:")
        try:
            result = await session.call_tool("get_rag_stats")
            stats = json.loads(result.content[0].text)
            if "error" in stats:
                print(f"  Error: {stats['error']}")
            else:
                s = stats["statistics"]
                print(f"  Documents: {s['total_documents']}")
                print(f"  Chunks: {s['total_chunks']}")
                print(f"  Storage: {s['storage_size_mb']} MB")
        except Exception as e:
            print(f"  Error getting stats: {e}")
    
    async def _list_documents(self, session):
        """List all documents in the system"""
        print("\n📚 Documents in System:")
        try:
            result = await session.call_tool("list_documents")
            docs = json.loads(result.content[0].text)
            if "error" in docs:
                print(f"  Error: {docs['error']}")
            elif docs["total_documents"] == 0:
                print("  No documents uploaded yet.")
            else:
                for doc in docs["documents"]:
                    print(f"  - {doc['name']} ({doc['chunk_count']} chunks)")
        except Exception as e:
            print(f"  Error listing documents: {e}")
    
    async def _demo_pdf_upload(self, session):
        """Demo PDF upload functionality"""
        print("\n📄 PDF Upload Demo:")
        
        # You can modify this path to point to an actual PDF file
        sample_pdf_path = input("Enter path to a PDF file (or press Enter to skip): ").strip()
        
        if not sample_pdf_path:
            print("  Skipping PDF upload demo (no file provided)")
            return
        
        if not Path(sample_pdf_path).exists():
            print(f"  File not found: {sample_pdf_path}")
            return
        
        try:
            result = await session.call_tool("upload_pdf", arguments={
                "file_path": sample_pdf_path,
                "document_name": Path(sample_pdf_path).stem
            })
            
            response = json.loads(result.content[0].text)
            if "error" in response:
                print(f"  Error: {response['error']}")
            else:
                print(f"  ✅ Success: {response['message']}")
                print(f"  Document ID: {response['document_id']}")
                print(f"  Chunks created: {response['chunks_created']}")
        except Exception as e:
            print(f"  Error uploading PDF: {e}")
    
    async def _demo_search(self, session):
        """Demo search functionality"""
        print("\n🔍 Search Demo:")
        
        # First check if we have any documents
        try:
            result = await session.call_tool("list_documents")
            docs = json.loads(result.content[0].text)
            
            if docs["total_documents"] == 0:
                print("  No documents to search. Please upload a PDF first.")
                return
            
            # Demo searches
            search_queries = [
                "What is the main topic?",
                "key findings",
                "methodology",
                "conclusion"
            ]
            
            for query in search_queries:
                print(f"\n  Query: '{query}'")
                try:
                    result = await session.call_tool("search_documents", arguments={
                        "query": query,
                        "top_k": 3
                    })
                    
                    search_results = json.loads(result.content[0].text)
                    if "error" in search_results:
                        print(f"    Error: {search_results['error']}")
                    else:
                        print(f"    Found {search_results['total_results']} results:")
                        for i, res in enumerate(search_results['results'][:2], 1):
                            print(f"    {i}. Score: {res['score']:.3f}")
                            print(f"       Document: {res['document_name']}")
                            print(f"       Text: {res['text'][:100]}...")
                except Exception as e:
                    print(f"    Error searching: {e}")
        
        except Exception as e:
            print(f"  Error in search demo: {e}")
    
    async def _demo_resources(self, session):
        """Demo resource functionality"""
        print("\n📦 Resources Demo:")

        # List available resources
        resources = await session.list_resources()
        print("  Available resources:")
        for resource in resources.resources:
            print(f"    - {resource.uri}: {resource.description}")
        
        # Try to read some resources
        resource_uris = ["rag://documents", "rag://stats"]
        
        for uri in resource_uris:
            try:
                print(f"\n  Reading resource: {uri}")
                content = await session.read_resource(uri)
                print(f"    Content: {content.contents[0].text[:200]}...")
            except Exception as e:
                print(f"    Error reading {uri}: {e}")
    
    async def _demo_prompts(self, session):
        """Demo prompt functionality"""
        print("\n💬 Prompts Demo:")

        # List available prompts
        prompts = await session.list_prompts()
        print("  Available prompts:")
        for prompt in prompts.prompts:
            print(f"    - {prompt.name}: {prompt.description}")
        
        # Demo RAG query prompt
        try:
            print(f"\n  Testing RAG query prompt:")
            prompt = await session.get_prompt("rag_query_prompt", arguments={
                "query": "What are the key findings?",
                "context_chunks": "Sample context: This document discusses important research findings about AI and machine learning applications."
            })
            print(f"    Generated prompt with {len(prompt.messages)} messages")
            for i, msg in enumerate(prompt.messages):
                print(f"    {i+1}. {msg.role}: {msg.content.text[:100]}...")
        except Exception as e:
            print(f"    Error with RAG prompt: {e}")

    async def interactive_mode(self):
        """Run interactive mode for testing"""
        async with sse_client(url=self.server_url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                
                print("\n🔧 Interactive RAG Client")
                print("Commands: upload, search, list, stats, quit")
                
                while True:
                    try:
                        command = input("\nrag> ").strip().lower()
                        
                        if command == "quit":
                            break
                        elif command == "upload":
                            file_path = input("PDF file path: ").strip()
                            if file_path and Path(file_path).exists():
                                result = await session.call_tool("upload_pdf", arguments={
                                    "file_path": file_path
                                })
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                            else:
                                print("Invalid file path")
                        
                        elif command == "search":
                            query = input("Search query: ").strip()
                            if query:
                                result = await session.call_tool("search_documents", arguments={
                                    "query": query,
                                    "top_k": 5
                                })
                                response = json.loads(result.content[0].text)
                                print(json.dumps(response, indent=2))
                        
                        elif command == "list":
                            result = await session.call_tool("list_documents")
                            response = json.loads(result.content[0].text)
                            print(json.dumps(response, indent=2))
                        
                        elif command == "stats":
                            result = await session.call_tool("get_rag_stats")
                            response = json.loads(result.content[0].text)
                            print(json.dumps(response, indent=2))
                        
                        else:
                            print("Unknown command. Available: upload, search, list, stats, quit")
                    
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"Error: {e}")


async def main():
    client = RAGClient()
    
    print("RAG System Client")
    print("1. Run demo")
    print("2. Interactive mode")
    
    choice = input("Choose mode (1 or 2): ").strip()
    
    if choice == "1":
        await client.run_demo()
    elif choice == "2":
        await client.interactive_mode()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())