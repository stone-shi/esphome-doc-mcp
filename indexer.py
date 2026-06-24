import os
import re
import yaml
import httpx
import sqlite3
import hashlib
import subprocess
from pathlib import Path
import config
from db import (
    init_db,
    get_connection,
    get_all_documents,
    store_document,
    delete_document_by_path,
    store_chunk
)

def parse_md_file(content):
    """
    Parses YAML frontmatter and body of a markdown/mdx file.
    Returns: (title, description, clean_body)
    """
    title = "Untitled"
    description = ""
    body = content
    
    # Try parsing frontmatter
    # Format:
    # ---
    # title: "Something"
    # ---
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if match:
        frontmatter_str = match.group(1)
        body = match.group(2)
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
            if isinstance(frontmatter, dict):
                title = frontmatter.get("title", title)
                description = frontmatter.get("description", "")
        except Exception:
            # Fallback regex if YAML parsing fails
            title_match = re.search(r"^title:\s*['\"]?(.*?)['\"]?\s*$", frontmatter_str, re.MULTILINE)
            if title_match:
                title = title_match.group(1)
            desc_match = re.search(r"^description:\s*['\"]?(.*?)['\"]?\s*$", frontmatter_str, re.MULTILINE)
            if desc_match:
                description = desc_match.group(1)
    
    return title, description, body

def chunk_text(body, max_chunk_len=1500):
    """
    Splits text by markdown headers and sub-chunks long sections if they exceed max_chunk_len.
    Returns a list of dicts: [{"header": str, "content": str}]
    """
    chunks = []
    
    # Matches lines like: ## My Section
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    
    # Find all header positions
    header_positions = []
    for m in header_pattern.finditer(body):
        header_positions.append({
            "start": m.start(),
            "end": m.end(),
            "level": len(m.group(1)),
            "text": m.group(2).strip()
        })
        
    if not header_positions:
        # No headers found, sub-chunk the entire body as "Introduction"
        return sub_chunk("Introduction", body, max_chunk_len)
        
    # Process section before the first header
    intro_text = body[:header_positions[0]["start"]].strip()
    if intro_text:
        chunks.extend(sub_chunk("Introduction", intro_text, max_chunk_len))
        
    # Process each header's section
    for i in range(len(header_positions)):
        curr_header = header_positions[i]
        start_idx = curr_header["end"]
        end_idx = header_positions[i+1]["start"] if i + 1 < len(header_positions) else len(body)
        
        section_content = body[start_idx:end_idx].strip()
        if section_content:
            chunks.extend(sub_chunk(curr_header["text"], section_content, max_chunk_len))
            
    return chunks

def sub_chunk(header, text, max_len):
    """
    Splits a text block into smaller sub-chunks by paragraph, falling back to sliding window.
    """
    if len(text) <= max_len:
        return [{"header": header, "content": text}]
        
    sub_chunks = []
    # Split by paragraphs
    paragraphs = text.split("\n\n")
    current_chunk = []
    current_len = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        if current_len + len(p) > max_len:
            if current_chunk:
                sub_chunks.append({
                    "header": header,
                    "content": "\n\n".join(current_chunk)
                })
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p) + 2 # account for double newline
            
    if current_chunk:
        sub_chunks.append({
            "header": header,
            "content": "\n\n".join(current_chunk)
        })
        
    # If any paragraph is still too long, perform a character-level split with overlap
    final_chunks = []
    for sc in sub_chunks:
        content = sc["content"]
        if len(content) <= max_len:
            final_chunks.append(sc)
        else:
            start = 0
            while start < len(content):
                end = start + max_len
                chunk_slice = content[start:end]
                final_chunks.append({
                    "header": header,
                    "content": chunk_slice
                })
                # Move start forward with 150 characters overlap
                start += max_len - 150
                
    return final_chunks

def get_embeddings(texts, api_base, model, api_key):
    """
    Sends a POST request to LiteLLM to retrieve text embeddings.
    """
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    url = f"{api_base}/embeddings"
    payload = {
        "model": model,
        "input": texts
    }
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        res_data = response.json()
        
        # Extract embeddings and sort by API output index to preserve order
        data = res_data.get("data", [])
        data.sort(key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data]

def sync_repository(repo_url, repo_path):
    """
    Clones the repository if missing, or pulls the latest main updates.
    """
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not (repo_path / ".git").exists():
        print(f"Cloning {repo_url} into {repo_path}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True
        )
    else:
        print(f"Pulling latest updates in {repo_path}...")
        # Clean any local changes first to prevent merge conflicts
        subprocess.run(
            ["git", "-C", str(repo_path), "reset", "--hard"],
            check=True
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "pull"],
            check=True
        )

def calculate_file_hash(filepath):
    """
    Calculates MD5 hash of file to identify modifications.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def run_indexing(force=False):
    """
    Clones/pulls the documentation and processes modified files.
    """
    db_path = config.DB_PATH
    repo_path = config.REPO_PATH
    repo_url = config.REPO_URL
    api_base = config.LITELLM_API_BASE
    model = config.LITELLM_MODEL
    api_key = config.LITELLM_API_KEY
    
    # 1. Initialize DB and open connection
    init_db(db_path)
    conn = get_connection(db_path)
    
    # 2. Synchronize Git repo
    try:
        sync_repository(repo_url, repo_path)
    except Exception as e:
        print(f"Git synchronization failed: {e}. Proceeding with existing local files.")
        
    if not repo_path.exists():
        print("Error: Local documentation repository path does not exist.")
        conn.close()
        return {"error": "Repository path not found"}
        
    # 3. Locate documentation content files
    # ESPHome docs repository structures them under src/content/docs
    docs_dir = repo_path / "src" / "content" / "docs"
    if not docs_dir.exists():
        # Fallback to scanning everything in the repo
        docs_dir = repo_path
        
    print(f"Scanning for documentation files in {docs_dir}...")
    all_files = []
    for ext in ["*.md", "*.mdx"]:
        all_files.extend(list(docs_dir.rglob(ext)))
        
    # Exclude system folders, node_modules, and standard guide files (like README.md)
    filtered_files = []
    for filepath in all_files:
        rel_path = filepath.relative_to(repo_path)
        rel_str = str(rel_path)
        
        # Skip hidden files/directories and node_modules
        if any(part.startswith(".") for part in rel_path.parts) or "node_modules" in rel_path.parts:
            continue
            
        # Skip top-level files that are not core docs
        if filepath.name.lower() in ["readme.md", "contributing.md", "changelog.md", "license.md"]:
            continue
            
        filtered_files.append((filepath, rel_str))
        
    print(f"Found {len(filtered_files)} documentation files in repository.")
    
    # 4. Get currently indexed files
    indexed_docs = get_all_documents(conn)
    active_filepaths = {rel_str for _, rel_str in filtered_files}
    
    # Delete removed files from DB
    deleted_count = 0
    for db_filepath in list(indexed_docs.keys()):
        if db_filepath not in active_filepaths:
            print(f"Deleting removed document from index: {db_filepath}")
            delete_document_by_path(conn, db_filepath)
            deleted_count += 1
            indexed_docs.pop(db_filepath)
            
    # Determine which files need indexing (new or modified)
    to_process = []
    for filepath, rel_str in filtered_files:
        try:
            curr_hash = calculate_file_hash(filepath)
        except Exception as e:
            print(f"Could not hash file {filepath}: {e}")
            continue
            
        is_new = rel_str not in indexed_docs
        is_modified = not is_new and indexed_docs[rel_str]["file_hash"] != curr_hash
        
        if is_new or is_modified or force:
            doc_id = indexed_docs.get(rel_str, {}).get("id")
            to_process.append((filepath, rel_str, curr_hash, is_new, doc_id))
            
    print(f"Indexing needed for {len(to_process)} documents.")
    
    added_count = 0
    updated_count = 0
    
    # 5. Embed and index documents
    for filepath, rel_str, curr_hash, is_new, doc_id in to_process:
        print(f"Processing: {rel_str} ...")
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            title, description, body = parse_md_file(content)
            
            # If doc exists, delete old entry to cascade delete old chunks
            if not is_new and doc_id:
                delete_document_by_path(conn, rel_str)
                
            # Create fresh document record
            new_doc_id = store_document(conn, rel_str, title, curr_hash)
            
            # Create semantic chunks
            chunks = chunk_text(body)
            if not chunks:
                chunks = [{"header": "Introduction", "content": description or "Document details."}]
                
            # Build list of metadata-enriched text blocks for embeddings
            chunk_texts_to_embed = []
            for chunk in chunks:
                context_text = f"Document: {title}\nSection: {chunk['header']}\nContent:\n{chunk['content']}"
                chunk_texts_to_embed.append(context_text)
                
            # Batch embedding API requests
            batch_size = 32
            embeddings = []
            for j in range(0, len(chunk_texts_to_embed), batch_size):
                batch_texts = chunk_texts_to_embed[j:j+batch_size]
                batch_embs = get_embeddings(batch_texts, api_base, model, api_key)
                embeddings.extend(batch_embs)
                
            # Insert chunks into database
            for idx, chunk in enumerate(chunks):
                store_chunk(conn, new_doc_id, idx, chunk["header"], chunk["content"], embeddings[idx])
                
            if is_new:
                added_count += 1
            else:
                updated_count += 1
                
        except Exception as e:
            print(f"Failed to process and index file {rel_str}: {e}")
            delete_document_by_path(conn, rel_str)
            
    conn.close()
    
    results = {
        "added": added_count,
        "updated": updated_count,
        "deleted": deleted_count
    }
    print(f"Sync complete. Results: {results}")
    return results

if __name__ == "__main__":
    import sys
    force_index = "--force" in sys.argv
    print("Running indexer command line sync...")
    run_indexing(force=force_index)
