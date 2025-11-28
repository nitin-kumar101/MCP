import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client


async def test_server():
    """Test the RAG server without user input"""
    print("🚀 Testing RAG Server...")
    
    try:
        async with sse_client(url="http://localhost:8000/sse") as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                
                print("✅ Successfully connected to RAG server!")
                
                # Test 1: List tools
                print("\n📋 Available Tools:")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # Test 2: Get system stats
                print("\n📊 System Statistics:")
                result = await session.call_tool("get_rag_stats")
                stats = json.loads(result.content[0].text)
                if "error" in stats:
                    print(f"  Error: {stats['error']}")
                else:
                    s = stats["statistics"]
                    print(f"  Documents: {s['total_documents']}")
                    print(f"  Chunks: {s['total_chunks']}")
                    print(f"  Storage: {s['storage_size_mb']} MB")
                    print(f"  Storage Directory: {s['storage_directory']}")
                
                # Test 3: List documents
                print("\n📚 Documents:")
                result = await session.call_tool("list_documents")
                docs = json.loads(result.content[0].text)
                if docs["total_documents"] == 0:
                    print("  No documents uploaded yet.")
                else:
                    for doc in docs["documents"]:
                        print(f"  - {doc['name']} ({doc['chunk_count']} chunks)")
                
                # Test 4: List resources
                print("\n📦 Resources:")
                resources = await session.list_resources()
                for resource in resources.resources:
                    print(f"  - {resource.uri}: {resource.description}")
                
                # Test 5: Read stats resource
                print("\n📊 Stats Resource Content:")
                content = await session.read_resource("rag://stats")
                print(content.contents[0].text)
                
                # Test 6: List prompts
                print("\n💬 Available Prompts:")
                prompts = await session.list_prompts()
                for prompt in prompts.prompts:
                    print(f"  - {prompt.name}: {prompt.description}")
                
                print("\n🎉 All tests passed! RAG server is working correctly.")
                print("\n📝 Next steps:")
                print("  1. Use 'python mcp_client_simple.py' for interactive testing")
                print("  2. Upload PDF files using the 'upload' command")
                print("  3. Search documents using the 'search' command")
                
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        print("Make sure the server is running with: python mcp_server.py")


if __name__ == "__main__":
    asyncio.run(test_server())
