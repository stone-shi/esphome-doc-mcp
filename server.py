import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.middleware.cors import CORSMiddleware

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.sse import SseServerTransport

import config
import db
import indexer

# Initialize FastMCP Server
mcp = FastMCP("ESPHome Doc Search Server")

# Disable DNS rebinding protection to allow connection from external hosts/IPs (e.g. 10.100.0.50:4008)
mcp.settings.transport_security.enable_dns_rebinding_protection = False

# Configure SSE Transport (Legacy)
# Note: Clients will connect to GET /sse to start the stream.
# The server will tell the client to post messages to POST /messages/.
transport = SseServerTransport("/messages/", security_settings=mcp.settings.transport_security)

# Configure Streamable HTTP app (Modern)
mcp.streamable_http_app()
streamable_app = StreamableHTTPASGIApp(mcp.session_manager)

# Declare the global background sync task tracker
background_sync_task = None

async def run_scheduled_sync():
    """
    Background loop to run document synchronization at configurable intervals.
    """
    # 1. On startup: If database has no chunks, index immediately.
    conn = db.get_connection(config.DB_PATH)
    try:
        stats = db.get_index_stats(conn)
        total_chunks = stats["total_chunks"]
    except Exception as e:
        print(f"Error checking database stats: {e}")
        total_chunks = 0
    finally:
        conn.close()
        
    if total_chunks == 0:
        print("Database is empty on startup. Starting initial indexing...")
        try:
            # Run indexing in a background thread to keep event loop free
            results = await asyncio.to_thread(indexer.run_indexing)
            print(f"Initial indexing complete: {results}")
        except Exception as e:
            print(f"Initial indexing failed: {e}")
            
    # 2. Infinite loop for periodic synchronization
    while True:
        try:
            print(f"Next background sync scheduled in {config.SYNC_INTERVAL_HOURS} hours.")
            await asyncio.sleep(config.SYNC_INTERVAL_HOURS * 3600)
            print("Triggering scheduled background documentation sync...")
            results = await asyncio.to_thread(indexer.run_indexing)
            print(f"Scheduled sync finished: {results}")
        except asyncio.CancelledError:
            print("Background sync task cancelled.")
            break
        except Exception as e:
            print(f"Error during scheduled background sync: {e}")
            # Cool-down sleep before trying again
            await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Lifecycle
    global background_sync_task
    
    # Initialize the Database Schema if missing
    db.init_db(config.DB_PATH)
    
    if config.SYNC_INTERVAL_HOURS > 0:
        background_sync_task = asyncio.create_task(run_scheduled_sync())
        print(f"Document Auto-Sync Scheduled: Every {config.SYNC_INTERVAL_HOURS} hours.")
    else:
        print("Document Auto-Sync is disabled (SYNC_INTERVAL_HOURS <= 0).")
        
    # Start the Streamable HTTP session manager context
    async with mcp.session_manager.run():
        yield
    
    # Shutdown Lifecycle
    if background_sync_task:
        background_sync_task.cancel()
        try:
            await background_sync_task
        except asyncio.CancelledError:
            pass
        print("Background auto-sync task cleaned up.")

# Initialize FastAPI App
fastapi_app = FastAPI(lifespan=lifespan)

# Allow CORS for MCP Client connections if needed
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- MCP SSE Handlers -----------------

@fastapi_app.get("/sse")
async def handle_sse(request: Request):
    """
    Accepts SSE connection requests from the client.
    """
    async with transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (in_stream, out_stream):
        await mcp._mcp_server.run(
            in_stream,
            out_stream,
            mcp._mcp_server.create_initialization_options()
        )

# Mount the messages handler under the matching endpoint
fastapi_app.mount("/messages/", transport.handle_post_message)

class MCPRoutingMiddleware:
    """
    ASGI middleware to route requests to the appropriate MCP transport.
    Distinguishes between legacy SSE and modern Streamable HTTP.
    """
    def __init__(self, app, sse_transport, streamable_app):
        self.app = app
        self.sse_transport = sse_transport
        self.streamable_app = streamable_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            method = scope["method"]
            headers = dict(scope.get("headers", []))
            has_mcp_session = b"mcp-session-id" in headers

            # Route Streamable HTTP requests (POST /sse, DELETE /sse, or GET /sse with session id header)
            if path in ("/sse", "/sse/"):
                if method == "POST" or method == "DELETE" or (method == "GET" and has_mcp_session):
                    await self.streamable_app(scope, receive, send)
                    return
                # Otherwise, GET /sse without session id is handled by FastAPI (legacy SSE client initiation)
                
            # Route legacy SSE POST messages (POST /messages or POST /messages/)
            elif path in ("/messages", "/messages/"):
                if method == "POST":
                    await self.sse_transport.handle_post_message(scope, receive, send)
                    return

        await self.app(scope, receive, send)

# Wrap the FastAPI app inside the routing middleware
app = MCPRoutingMiddleware(fastapi_app, transport, streamable_app)

# ----------------- MCP Exposed Tools -----------------

@mcp.tool()
async def search_docs(query: str, limit: int = 5) -> str:
    """
    Semantic search over the ESPHome documentation using LiteLLM embeddings.
    Returns the top matching text snippets with file paths and similarity scores.
    """
    if not query.strip():
        return "Please provide a valid query."
        
    try:
        # 1. Embed query
        query_vectors = await asyncio.to_thread(
            indexer.get_embeddings,
            [query],
            config.LITELLM_API_BASE,
            config.LITELLM_MODEL,
            config.LITELLM_API_KEY
        )
        
        if not query_vectors:
            return "Failed to generate query embedding."
            
        query_vector = query_vectors[0]
        
        # 2. Query SQLite DB using cosine similarity
        conn = db.get_connection(config.DB_PATH)
        try:
            results = db.search_similar_chunks(conn, query_vector, limit=limit)
        finally:
            conn.close()
            
        if not results:
            return "No matching documentation found. Try running sync_docs to index documents first."
            
        # 3. Format results beautifully
        formatted_output = []
        for idx, res in enumerate(results, 1):
            formatted_output.append(
                f"Match #{idx}\n"
                f"File: {res['filepath']}\n"
                f"Title: {res['title']}\n"
                f"Section: {res['header']}\n"
                f"Similarity Score: {res['score']:.4f}\n"
                f"----------------------------------------\n"
                f"{res['content']}\n"
                f"========================================\n"
            )
            
        return "\n".join(formatted_output)
        
    except Exception as e:
        return f"Error executing semantic search: {e}"

@mcp.tool()
def get_doc_page(path: str) -> str:
    """
    Retrieves the raw Markdown / MDX contents of a specific documentation file.
    Use this to read full details and config parameters of components.
    Argument 'path' must be a relative file path in the repository (e.g. 'src/content/docs/components/sensor/wifi.mdx').
    """
    if not path.strip():
        return "Error: Path cannot be empty."
        
    repo_root = config.REPO_PATH.resolve()
    requested_path = (repo_root / path).resolve()
    
    # Path traversal vulnerability validation
    if not requested_path.is_relative_to(repo_root):
        return "Error: Access denied. Path must be inside the documentation repository."
        
    if not requested_path.exists():
        return f"Error: Documentation file '{path}' not found."
        
    try:
        with open(requested_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading document file: {e}"

@mcp.tool()
async def sync_docs() -> str:
    """
    Manually triggers git pull and incremental documentation indexing.
    Returns stats on added, updated, and deleted files.
    """
    try:
        results = await asyncio.to_thread(indexer.run_indexing)
        if "error" in results:
            return f"Synchronization failed: {results['error']}"
            
        return (
            "Documentation synchronized successfully!\n"
            f"- Added documents: {results['added']}\n"
            f"- Updated documents: {results['updated']}\n"
            f"- Deleted documents: {results['deleted']}"
        )
    except Exception as e:
        return f"Error running manual synchronization: {e}"

@mcp.tool()
def get_index_status() -> str:
    """
    Returns statistics about the current documentation index database.
    """
    conn = db.get_connection(config.DB_PATH)
    try:
        stats = db.get_index_stats(conn)
        return (
            "ESPHome Documentation Index Status:\n"
            f"- Total Documents Indexed: {stats['total_documents']}\n"
            f"- Total Chunks Created: {stats['total_chunks']}\n"
            f"- Last Indexed Timestamp: {stats['last_indexed']}"
        )
    except Exception as e:
        return f"Error reading database index status: {e}"
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    print("Starting ESPHome Doc Search MCP SSE Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
