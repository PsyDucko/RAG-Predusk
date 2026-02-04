# main.py
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# RAG imports
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# PDF processing - simplified approach
import pdfplumber
from pypdf import PdfReader
import PyPDF2
from io import BytesIO
import re

# ============================================================================
# PDF Processing Classes
# ============================================================================

class PDFContentExtractor:
    """Extract text from PDFs using robust methods"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pages_content = []
    
    def extract_all(self) -> Dict:
        """Extract all content from PDF"""
        print(f"📄 Processing: {Path(self.pdf_path).name}")
        
        # Try multiple extraction methods
        methods = [
            self._extract_with_pypdf,
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
            self._extract_binary_fallback
        ]
        
        for method in methods:
            try:
                print(f"   Trying {method.__name__}...")
                result = method()
                if result and len(result['pages']) > 0 and any(p['text'].strip() for p in result['pages']):
                    self.pages_content = result['pages']
                    print(f"   ✅ Success with {method.__name__}")
                    break
            except Exception as e:
                print(f"   ❌ {method.__name__} failed: {str(e)[:50]}")
                continue
        
        if not self.pages_content or not any(p['text'].strip() for p in self.pages_content):
            print(f"   ⚠️ All extraction methods failed or produced no content")
            # Create a minimal document to indicate the file exists
            self.pages_content = [{
                'page_number': 1,
                'text': f"[PDF file: {Path(self.pdf_path).name} - Could not extract text content]",
                'tables': []
            }]
        
        print(f"✅ Processed {len(self.pages_content)} pages")
        return self._compile_results()
    
    def _extract_with_pypdf(self) -> Dict:
        """Extract using pypdf (most reliable)"""
        try:
            with open(self.pdf_path, 'rb') as file:
                # Read file to check if it's valid
                header = file.read(5)
                file.seek(0)
                
                if header[:4] != b'%PDF':
                    raise ValueError("Not a valid PDF file")
                
                reader = PdfReader(file)
                pages = []
                
                for page_num in range(len(reader.pages)):
                    page_data = {
                        'page_number': page_num + 1,
                        'text': '',
                        'tables': []
                    }
                    
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            # Clean up the text
                            text = self._clean_text(text)
                            page_data['text'] = text
                    except Exception as e:
                        print(f"      Page {page_num + 1} error: {e}")
                        continue
                    
                    pages.append(page_data)
                
                return {'pages': pages}
        except Exception as e:
            raise Exception(f"PyPDF failed: {e}")
    
    def _extract_with_pdfplumber(self) -> Dict:
        """Extract using pdfplumber"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                pages = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_data = {
                        'page_number': page_num,
                        'text': '',
                        'tables': []
                    }
                    
                    try:
                        text = page.extract_text()
                        if text:
                            text = self._clean_text(text)
                            page_data['text'] = text
                    except:
                        pass
                    
                    pages.append(page_data)
                
                return {'pages': pages}
        except Exception as e:
            raise Exception(f"pdfplumber failed: {e}")
    
    def _extract_with_pypdf2(self) -> Dict:
        """Extract using PyPDF2 (legacy)"""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                pages = []
                
                for page_num in range(len(reader.pages)):
                    page_data = {
                        'page_number': page_num + 1,
                        'text': '',
                        'tables': []
                    }
                    
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            text = self._clean_text(text)
                            page_data['text'] = text
                    except:
                        pass
                    
                    pages.append(page_data)
                
                return {'pages': pages}
        except Exception as e:
            raise Exception(f"PyPDF2 failed: {e}")
    
    def _extract_binary_fallback(self) -> Dict:
        """Extract text using binary reading as last resort"""
        try:
            with open(self.pdf_path, 'rb') as file:
                content = file.read()
                
                # Try to extract any ASCII text
                text_chunks = []
                current_chunk = []
                
                for byte in content:
                    if 32 <= byte <= 126:  # Printable ASCII
                        current_chunk.append(chr(byte))
                    elif current_chunk:
                        chunk = ''.join(current_chunk)
                        # Only keep chunks that look like real text
                        if len(chunk) > 20 and ' ' in chunk:
                            text_chunks.append(chunk)
                        current_chunk = []
                
                if current_chunk:
                    chunk = ''.join(current_chunk)
                    if len(chunk) > 20 and ' ' in chunk:
                        text_chunks.append(chunk)
                
                if text_chunks:
                    full_text = ' '.join(text_chunks)
                    full_text = self._clean_text(full_text)
                    return {
                        'pages': [{
                            'page_number': 1,
                            'text': full_text[:10000],  # Limit size
                            'tables': []
                        }]
                    }
                else:
                    raise Exception("No readable text found")
        except Exception as e:
            raise Exception(f"Binary extraction failed: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # Remove control chars
        
        return text.strip()
    
    def _compile_results(self) -> Dict:
        return {
            'pages': self.pages_content,
            'total_pages': len(self.pages_content)
        }
    
    def create_documents(self) -> List[Dict]:
        """Create document chunks"""
        documents = []
        
        for page_data in self.pages_content:
            if page_data['text'] and page_data['text'].strip():
                documents.append({
                    'content': page_data['text'],
                    'metadata': {
                        'source': self.pdf_path,
                        'page': page_data['page_number'],
                        'has_tables': False
                    }
                })
        
        return documents

class TextProcessor:
    """Process text files"""
    
    @staticmethod
    def process_text_file(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Process text file into chunks"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                        break
                except:
                    continue
            
            if content is None:
                # Binary read as last resort
                with open(file_path, 'rb') as f:
                    binary = f.read()
                    content = binary.decode('ascii', errors='ignore')
        
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        if not content or not content.strip():
            return []
        
        # Clean content
        content = ' '.join(content.split())
        
        # Split into chunks
        words = content.split()
        documents = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                documents.append({
                    'content': chunk,
                    'metadata': {
                        'source': file_path,
                        'page': (i // (chunk_size - overlap)) + 1,
                        'chunk': (i // (chunk_size - overlap)) + 1
                    }
                })
        
        return documents
    
    @staticmethod
    def process_text_content(text: str, source_name: str = "user_input") -> List[Dict]:
        """Process raw text content"""
        if not text or not text.strip():
            return []
        
        # Clean text
        text = ' '.join(text.split())
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        documents = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                documents.append({
                    'content': paragraph,
                    'metadata': {
                        'source': source_name,
                        'page': i + 1
                    }
                })
        
        return documents

# ============================================================================
# RAG System Class
# ============================================================================

class RAGSystem:
    """Complete RAG system with vector store and LLM"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.llm_pipeline = None
        self.uploaded_files = []
        self.temp_dir = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the RAG system"""
        print("🚀 Initializing RAG System...")
        
        # Create temp directory for uploaded files
        self.temp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
        print(f"   Temp directory: {self.temp_dir}")
        
        # Load embedding model
        print("   Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"   ✅ Embedding model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB
        print("   Initializing vector store...")
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            
            # Try to load existing collection or create new
            try:
                self.collection = self.chroma_client.get_collection(name="rag_collection")
                print(f"   ✅ Loaded existing vector store: {self.collection.count()} documents")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="rag_collection",
                    metadata={"hnsw:space": "cosine"}
                )
                print("   ✅ Created new vector store")
        except Exception as e:
            print(f"   ❌ Failed to initialize vector store: {e}")
            raise
        
        # Load LLM (optional)
        print("   Loading LLM (optional)...")
        self.llm_pipeline = None  # We'll use simple concatenation for now
        print("   ⚠️  LLM disabled - using simple text concatenation")
        
        self.initialized = True
        print("✅ RAG System initialized!\n")
    
    def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temp directory"""
        import time
        
        # Create safe filename
        original_name = file.filename or "uploaded_file"
        safe_name = "".join(c for c in original_name if c.isalnum() or c in ('.', '-', '_', ' ')).strip()
        if not safe_name:
            safe_name = f"file_{int(time.time())}"
        
        # Ensure unique filename
        base_name, ext = os.path.splitext(safe_name)
        counter = 1
        final_name = safe_name
        
        while os.path.exists(os.path.join(self.temp_dir, final_name)):
            final_name = f"{base_name}_{counter}{ext}"
            counter += 1
        
        file_path = os.path.join(self.temp_dir, final_name)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            self.uploaded_files.append({
                'filename': final_name,
                'path': file_path,
                'size': os.path.getsize(file_path)
            })
            return file_path
        else:
            raise Exception(f"Failed to save file: {final_name}")
    
    def process_pdf_file(self, file_path: str) -> List[Dict]:
        """Process PDF file and return document chunks"""
        print(f"   Processing PDF: {Path(file_path).name}")
        
        try:
            # First verify it's a readable file
            if not os.path.exists(file_path):
                raise Exception("File does not exist")
            
            if os.path.getsize(file_path) == 0:
                raise Exception("File is empty")
            
            extractor = PDFContentExtractor(file_path)
            results = extractor.extract_all()
            documents = extractor.create_documents()
            
            if not documents:
                print(f"   ⚠️  No documents created from PDF")
                # Create a placeholder document
                documents = [{
                    'content': f"File: {Path(file_path).name} (PDF - content extraction failed)",
                    'metadata': {
                        'source': file_path,
                        'page': 1,
                        'has_tables': False
                    }
                }]
            
            print(f"   Created {len(documents)} document(s)")
            return documents
            
        except Exception as e:
            print(f"   ❌ PDF processing error: {e}")
            # Return minimal document
            return [{
                'content': f"File: {Path(file_path).name} (Error: {str(e)[:100]})",
                'metadata': {
                    'source': file_path,
                    'page': 1,
                    'has_tables': False
                }
            }]
    
    def process_text_file(self, file_path: str) -> List[Dict]:
        """Process text file and return document chunks"""
        print(f"   Processing text file: {Path(file_path).name}")
        
        try:
            documents = TextProcessor.process_text_file(file_path)
            print(f"   Created {len(documents)} document(s)")
            return documents
        except Exception as e:
            print(f"   ❌ Text file processing error: {e}")
            return []
    
    def process_text_content(self, text: str, source_name: str = "user_text") -> List[Dict]:
        """Process raw text content"""
        try:
            return TextProcessor.process_text_content(text, source_name)
        except Exception as e:
            print(f"Error processing text content: {e}")
            return []
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or not text.strip():
            return []
        
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_documents_to_store(self, documents: List[Dict]) -> Dict:
        """Add documents to vector store"""
        if not self.collection:
            raise Exception("Vector store not initialized")
        
        if not documents:
            return {
                'added_count': 0,
                'total_documents': self.collection.count(),
                'message': 'No documents to add'
            }
        
        # Filter and clean documents
        valid_documents = []
        for doc in documents:
            if doc.get('content'):
                content = doc['content'].strip()
                if content and len(content) >= 10:  # Minimum 10 chars
                    # Clean up content
                    content = ' '.join(content.split())
                    valid_documents.append({
                        'content': content,
                        'metadata': doc.get('metadata', {})
                    })
        
        if not valid_documents:
            return {
                'added_count': 0,
                'total_documents': self.collection.count(),
                'message': 'No valid content'
            }
        
        texts = [doc['content'] for doc in valid_documents]
        metadatas = [doc['metadata'] for doc in valid_documents]
        
        try:
            print(f"   Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                batch_size=8,
                convert_to_numpy=True
            )
            
            # Add to collection
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(texts))]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"   ✅ Added {len(texts)} documents")
            return {
                'added_count': len(texts),
                'total_documents': self.collection.count()
            }
            
        except Exception as e:
            print(f"   ❌ Error adding to vector store: {e}")
            return {
                'added_count': 0,
                'total_documents': self.collection.count(),
                'error': str(e)
            }
    
    def clear_vector_store(self) -> Dict:
        """Clear all documents from vector store"""
        try:
            self.chroma_client.delete_collection(name="rag_collection")
            self.collection = self.chroma_client.create_collection(
                name="rag_collection",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Clear temp files
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
            self.uploaded_files = []
            
            return {
                'success': True,
                'message': 'Vector store cleared successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def mmr_retrieval(self, query: str, k: int = 5, lambda_param: float = 0.5) -> List[Dict]:
        """Maximal Marginal Relevance retrieval"""
        if not self.collection or self.collection.count() == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query)
            fetch_k = min(20, self.collection.count())
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=fetch_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                return []
            
            # Simple retrieval (skip MMR for now)
            retrieved_docs = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                if i >= k:
                    break
                
                similarity = 1 - (distance ** 2 / 2)
                retrieved_docs.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance': similarity
                })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def query(self, 
              question: str, 
              k: int = 4, 
              lambda_param: float = 0.5,
              include_sources: bool = True) -> Dict:
        """Main query function"""
        if not self.initialized:
            return {
                "answer": "System not ready. Please try again.",
                "sources": [],
                "document_count": 0,
                "total_documents": 0
            }
        
        if not self.collection or self.collection.count() == 0:
            return {
                "answer": "No documents in knowledge base. Please upload files first.",
                "sources": [],
                "document_count": 0,
                "total_documents": 0
            }
        
        # Retrieve documents
        retrieved_docs = self.mmr_retrieval(question, k, lambda_param)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "document_count": 0,
                "total_documents": self.collection.count()
            }
        
        # Create simple answer from retrieved documents
        answer_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = Path(doc['metadata'].get('source', 'Unknown')).name
            preview = doc['content']
            if len(preview) > 300:
                preview = preview[:300] + "..."
            
            answer_parts.append(f"{i}. From {source}: {preview}")
        
        answer = "Based on the documents:\n\n" + "\n\n".join(answer_parts)
        
        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "document_count": len(retrieved_docs),
            "total_documents": self.collection.count()
        }
        
        if include_sources:
            sources = []
            for i, doc in enumerate(retrieved_docs, 1):
                source_name = Path(doc['metadata'].get('source', 'Unknown')).name
                preview = doc['content']
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                
                sources.append({
                    "id": i,
                    "source": source_name,
                    "page": doc['metadata'].get('page', '?'),
                    "relevance": round(doc.get('relevance', 0), 3),
                    "preview": preview
                })
            response["sources"] = sources
        
        return response
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'initialized': self.initialized,
            'vector_store_documents': self.collection.count() if self.collection else 0,
            'llm_available': False,  # Simple mode
            'uploaded_files_count': len(self.uploaded_files),
            'uploaded_files': [
                {
                    'filename': f['filename'],
                    'size': f'{f["size"] / 1024:.1f} KB'
                } for f in self.uploaded_files[-5:]
            ],
            'embedding_model': 'all-MiniLM-L6-v2'
        }

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Free RAG System API",
    description="Simple RAG Pipeline with PDF Upload",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(persist_dir="./chroma_db")

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    k: int = 4
    lambda_param: float = 0.5
    include_sources: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    document_count: int
    total_documents: int
    sources: Optional[List[Dict]] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    document_count: Optional[int] = None
    total_documents: Optional[int] = None
    failed_files: Optional[List[str]] = None

class SystemInfoResponse(BaseModel):
    initialized: bool
    vector_store_documents: int
    llm_available: bool
    uploaded_files_count: int
    uploaded_files: List[Dict[str, Any]]
    embedding_model: str

# HTML Interface (simplified)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple RAG System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .tabs {
            display: flex;
            margin: 20px 0;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
        }
        .tab.active {
            background: white;
            border-color: #ddd;
            border-bottom-color: white;
            border-radius: 5px 5px 0 0;
        }
        .tab-content {
            display: none;
            padding: 20px 0;
        }
        .tab-content.active {
            display: block;
        }
        .upload-area {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background: #f9f9f9;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 5px;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
            min-height: 100px;
        }
        .result {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .success { color: #4CAF50; background: #e8f5e9; padding: 10px; border-radius: 5px; }
        .error { color: #f44336; background: #ffebee; padding: 10px; border-radius: 5px; }
        .info { color: #2196F3; background: #e3f2fd; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Simple RAG System</h1>
        
        <div class="info" id="system-info">Loading...</div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('upload')">📤 Upload</div>
            <div class="tab" onclick="switchTab('query')">❓ Query</div>
        </div>
        
        <div id="upload-tab" class="tab-content active">
            <h2>Upload Files</h2>
            <div class="upload-area">
                <input type="file" id="file-input" multiple style="display: none;">
                <button onclick="document.getElementById('file-input').click()">Choose Files</button>
                <p>or drag & drop files here</p>
                <p><small>Supported: PDF, TXT, MD files</small></p>
            </div>
            <div id="file-list"></div>
            <button onclick="uploadFiles()" id="upload-btn">Upload</button>
            <button onclick="clearFiles()">Clear</button>
            <div id="upload-result"></div>
        </div>
        
        <div id="query-tab" class="tab-content">
            <h2>Ask Questions</h2>
            <textarea id="question" placeholder="Ask a question about your documents..."></textarea>
            <button onclick="askQuestion()" id="ask-btn">Ask</button>
            <div id="query-result" class="result" style="display: none;">
                <h3>Answer:</h3>
                <div id="answer"></div>
                <h3>Sources:</h3>
                <div id="sources"></div>
            </div>
        </div>
    </div>
    
    <script>
        let files = [];
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName + '-tab').style.display = 'block';
            document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
        }
        
        document.getElementById('file-input').addEventListener('change', function(e) {
            files = Array.from(e.target.files);
            updateFileList();
        });
        
        function updateFileList() {
            const list = document.getElementById('file-list');
            list.innerHTML = files.map(f => 
                `<div>${f.name} (${(f.size/1024).toFixed(1)} KB)</div>`
            ).join('');
            document.getElementById('upload-btn').disabled = files.length === 0;
        }
        
        function clearFiles() {
            files = [];
            updateFileList();
            document.getElementById('upload-result').innerHTML = '';
        }
        
        async function uploadFiles() {
            const btn = document.getElementById('upload-btn');
            btn.disabled = true;
            btn.textContent = 'Uploading...';
            
            const formData = new FormData();
            files.forEach(f => formData.append('files', f));
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                const display = document.getElementById('upload-result');
                if (result.success) {
                    display.innerHTML = `<div class="success">${result.message}<br>Added ${result.document_count} chunks</div>`;
                    files = [];
                    updateFileList();
                    updateSystemInfo();
                } else {
                    display.innerHTML = `<div class="error">${result.message}</div>`;
                }
            } catch (error) {
                document.getElementById('upload-result').innerHTML = 
                    `<div class="error">Upload failed: ${error.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Upload';
            }
        }
        
        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) return alert('Enter a question');
            
            const btn = document.getElementById('ask-btn');
            btn.disabled = true;
            btn.textContent = 'Searching...';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question, k: 4})
                });
                const result = await response.json();
                
                document.getElementById('answer').innerHTML = result.answer.replace(/\\n/g, '<br>');
                if (result.sources) {
                    const sourcesHtml = result.sources.map(s => 
                        `<div><strong>${s.source}</strong> (relevance: ${s.relevance})<br>${s.preview}</div><hr>`
                    ).join('');
                    document.getElementById('sources').innerHTML = sourcesHtml;
                }
                document.getElementById('query-result').style.display = 'block';
            } catch (error) {
                document.getElementById('answer').innerHTML = 
                    `<div class="error">Error: ${error.message}</div>`;
                document.getElementById('query-result').style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Ask';
            }
        }
        
        async function updateSystemInfo() {
            try {
                const response = await fetch('/system/info');
                const info = await response.json();
                document.getElementById('system-info').innerHTML = 
                    `📊 Documents: ${info.vector_store_documents} | 📁 Files: ${info.uploaded_files_count}`;
            } catch (error) {
                console.error('Failed to update system info:', error);
            }
        }
        
        // Initialize
        updateSystemInfo();
        setInterval(updateSystemInfo, 10000); // Update every 10 seconds
        
        // Allow Enter key in textarea
        document.getElementById('question').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    try:
        rag_system.initialize()
    except Exception as e:
        print(f"⚠️  Initialization error: {e}")

@app.get("/")
async def read_root():
    return HTMLResponse(content=html_content)

@app.get("/system/info")
async def system_info():
    try:
        info = rag_system.get_system_info()
        return info
    except:
        return {"initialized": False}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        if not files:
            return {"success": False, "message": "No files provided"}
        
        all_docs = []
        processed = []
        failed = []
        
        for file in files:
            try:
                # Save file
                file_path = rag_system.save_uploaded_file(file)
                
                # Process based on type
                if file.filename.lower().endswith('.pdf'):
                    docs = rag_system.process_pdf_file(file_path)
                elif file.filename.lower().endswith(('.txt', '.md', '.csv')):
                    docs = rag_system.process_text_file(file_path)
                else:
                    failed.append(f"{file.filename}: Unsupported type")
                    continue
                
                if docs:
                    all_docs.extend(docs)
                    processed.append(file.filename)
                else:
                    failed.append(f"{file.filename}: No content")
                    
            except Exception as e:
                failed.append(f"{file.filename}: {str(e)[:50]}")
                continue
        
        if not all_docs:
            return {
                "success": False,
                "message": "Could not extract content from any files",
                "failed_files": failed
            }
        
        # Add to vector store
        result = rag_system.add_documents_to_store(all_docs)
        
        if 'error' in result:
            return {
                "success": False,
                "message": f"Storage error: {result['error']}",
                "failed_files": failed
            }
        
        msg = f"Processed {len(processed)} files"
        if failed:
            msg += f" ({len(failed)} failed)"
        
        return {
            "success": True,
            "message": msg,
            "document_count": result['added_count'],
            "total_documents": result['total_documents'],
            "failed_files": failed if failed else None
        }
        
    except Exception as e:
        return {"success": False, "message": f"Upload error: {str(e)}"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        result = rag_system.query(
            question=request.question,
            k=request.k,
            lambda_param=request.lambda_param,
            include_sources=request.include_sources
        )
        return result
    except Exception as e:
        return {
            "question": request.question,
            "answer": f"Error: {str(e)}",
            "document_count": 0,
            "total_documents": 0
        }

@app.post("/clear")
async def clear_documents():
    try:
        result = rag_system.clear_vector_store()
        return result
    except Exception as e:
        return {"success": False, "message": str(e)}

# ============================================================================
# Run the Application
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("🚀 Simple RAG System Starting...")
    print(f"{'='*60}")
    print(f"🌐 Web UI: http://{args.host}:{args.port}")
    print(f"💾 Storage: ./chroma_db")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)