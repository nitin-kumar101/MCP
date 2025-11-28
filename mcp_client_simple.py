import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from mcp.client.sse import sse_client


async def test_rag_server():
    """Simple test of the RAG server"""
    print("🚀 Testing RAG Server Connection...")
    
    try:
        async with sse_client(url="http://localhost:8000/sse") as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                
                print("✅ Connected to RAG server!")
                
                # 1. List available tools
                print("\n📋 Available Tools:")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # 2. Get system stats
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
                
                # 3. List documents
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
                
                # 4. Test PDF upload (optional)
                pdf_path = input("\n📄 Enter path to a PDF file to test upload (or press Enter to skip): ").strip()
                if pdf_path and Path(pdf_path).exists():
                    try:
                        print("  Uploading PDF...")
                        result = await session.call_tool("upload_pdf", arguments={
                            "file_path": pdf_path,
                            "document_name": Path(pdf_path).stem
                        })
                        
                        response = json.loads(result.content[0].text)
                        if "error" in response:
                            print(f"  ❌ Error: {response['error']}")
                        else:
                            print(f"  ✅ Success: {response['message']}")
                            print(f"  Document ID: {response['document_id']}")
                            print(f"  Chunks created: {response['chunks_created']}")
                            
                            # Test search after upload
                            print("\n🔍 Testing search...")
                            search_result = await session.call_tool("search_documents", arguments={
                                "query": "main topic",
                                "top_k": 3
                            })
                            
                            search_data = json.loads(search_result.content[0].text)
                            if "error" in search_data:
                                print(f"  Search error: {search_data['error']}")
                            else:
                                print(f"  Found {search_data['total_results']} results:")
                                for i, res in enumerate(search_data['results'][:2], 1):
                                    print(f"    {i}. Score: {res['score']:.3f}")
                                    print(f"       Text: {res['text'][:100]}...")
                    except Exception as e:
                        print(f"  Error uploading PDF: {e}")
                elif pdf_path:
                    print(f"  File not found: {pdf_path}")
                else:
                    print("  Skipping PDF upload test")
                
                # 5. Test resources
                print("\n📦 Testing Resources:")
                try:
                    resources = await session.list_resources()
                    print("  Available resources:")
                    for resource in resources.resources:
                        print(f"    - {resource.uri}")
                    
                    # Try reading stats resource
                    content = await session.read_resource("rag://stats")
                    print(f"\n  Stats resource content:")
                    print(f"    {content.contents[0].text}")
                except Exception as e:
                    print(f"  Error with resources: {e}")
                
                print("\n🎉 RAG Server Test Complete!")
                
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("Make sure the server is running with: python mcp_server.py")


async def interactive_mode():
    """Interactive mode for testing RAG functionality"""
    print("🔧 Interactive RAG Client")
    print("Commands: upload, search, list, stats, quit")
    
    try:
        async with sse_client(url="http://localhost:8000/sse") as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                print("✅ Connected to server!")
                
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
    
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")


async def main():
    print("RAG System Client")
    print("1. Run connection test")
    print("2. Interactive mode")
    
    choice = input("Choose mode (1 or 2): ").strip()
    
    if choice == "1":
        await test_rag_server()
    elif choice == "2":
        await interactive_mode()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
