"""
RAG (Retrieval-Augmented Generation) helper module for local LLM integration.
This module handles text chunking, embedding, and querying with a local LLM.
"""

import os
import sys
import subprocess
import time
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Embedding and vector storage
from sentence_transformers import SentenceTransformer
import faiss

# Text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

def install_packages():
    """Install required packages for RAG functionality."""
    packages = [
        "sentence-transformers",
        "faiss-cpu",  # Use faiss-gpu if GPU is available
        "langchain",
        "llama-cpp-python",  # For local LLM inference
    ]
    
    all_installed = True
    for package in packages:
        try:
            if package == "faiss-cpu":
                import faiss
                print(f"✅ {package} already installed.")
            else:
                module_name = package.replace("-", "_")
                __import__(module_name)
                print(f"✅ {package} already installed.")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully.")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")
                all_installed = False
    
    return all_installed

def download_model_if_needed(model_path="./models"):
    """Download models if they don't exist locally."""
    # Create models directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Check for embedding model
    embedding_model_path = os.path.join(model_path, "embedding_model")
    if not os.path.exists(embedding_model_path):
        print("📥 Downloading embedding model...")
        # This will trigger the download of the model when we initialize it
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=embedding_model_path)
        print("✅ Embedding model downloaded.")
    
    # Check for LLM model
    llm_model_path = os.path.join(model_path, "llama-2-7b-chat.Q4_K_M.gguf")
    if not os.path.exists(llm_model_path):
        print("⚠️ LLM model not found at:", llm_model_path)
        print("Please download a GGUF model file (llama-2-7b-chat.Q4_K_M.gguf recommended)")
        print("from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main")
        print("and place it in the './models/' directory")
        return False
    
    return True

class RAGProcessor:
    def __init__(self, model_path="./models"):
        """Initialize the RAG processor with embedding model and vector store."""
        self.model_path = model_path
        
        # Load embedding model
        embedding_model_path = os.path.join(model_path, "embedding_model")
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=embedding_model_path)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded (dimension: {self.embedding_dim})")
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize FAISS index (will be created per document)
        self.index = None
        self.chunks = []
        
    def process_transcript(self, transcript_text: str) -> bool:
        """Process transcript text into chunks and create embeddings index."""
        try:
            print("🔪 Chunking transcript text...")
            self.chunks = self.text_splitter.split_text(transcript_text)
            print(f"✅ Created {len(self.chunks)} chunks")
            
            # Create embeddings for chunks
            print("🧠 Creating embeddings...")
            embeddings = self.embedding_model.encode(self.chunks)
            
            # Create FAISS index
            print("📊 Creating vector index...")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(np.array(embeddings).astype('float32'))
            
            print("✅ RAG processing complete")
            return True
            
        except Exception as e:
            print(f"❌ Error processing transcript: {str(e)}")
            return False
            
    def save_index(self, file_path: str) -> bool:
        """Save index and chunks to disk."""
        try:
            # Save index
            index_path = f"{file_path}.index"
            faiss.write_index(self.index, index_path)
            
            # Save chunks
            chunks_path = f"{file_path}.chunks.json"
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Saved index to {index_path} and chunks to {chunks_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving index: {str(e)}")
            return False
            
    def load_index(self, file_path: str) -> bool:
        """Load index and chunks from disk."""
        try:
            # Load index
            index_path = f"{file_path}.index"
            self.index = faiss.read_index(index_path)
            
            # Load chunks
            chunks_path = f"{file_path}.chunks.json"
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
                
            print(f"✅ Loaded index from {index_path} and chunks from {chunks_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading index: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the most relevant chunks for a query."""
        if self.index is None or len(self.chunks) == 0:
            print("❌ No index available. Process a transcript first.")
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Get relevant chunks
        relevant_chunks = [self.chunks[idx] for idx in indices[0]]
        return relevant_chunks

class LocalLLM:
    def __init__(self, model_path="./models"):
        """Initialize the local LLM."""
        self.model_path = model_path
        self.llm = None
        
    def load_model(self, model_name="llama-2-7b-chat.Q4_K_M.gguf"):
        """Load the LLM model."""
        try:
            from llama_cpp import Llama
            
            full_model_path = os.path.join(self.model_path, model_name)
            if not os.path.exists(full_model_path):
                print(f"❌ Model not found at {full_model_path}")
                return False
                
            print(f"🔄 Loading LLM from {full_model_path}...")
            self.llm = Llama(
                model_path=full_model_path,
                n_ctx=2048,  # Context window size
                n_batch=512,  # Batch size for prompt processing
                n_gpu_layers=-1  # Attempt to offload all layers to GPU
            )
            print("✅ LLM loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading LLM: {str(e)}")
            return False
    
    def generate_response(self, query: str, context_chunks: List[str], max_tokens: int = 512) -> str:
        """Generate a response using the LLM with context chunks."""
        if self.llm is None:
            print("❌ LLM not loaded. Call load_model() first.")
            return "Error: LLM not loaded. Please load the model first."
        
        try:
            # Create a prompt with the context chunks
            context_text = "\n\n".join(context_chunks)
            
            prompt = f"""Below is a section of a transcript from a video:

{context_text}

Based on the above transcript, please answer the following question:
{query}

Answer:"""
            
            print(f"🤖 Generating response with {len(context_chunks)} context chunks...")
            
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["Human:", "\n\n\n"],
                echo=False
            )
            
            answer = response["choices"][0]["text"].strip()
            return answer
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

# Test function
if __name__ == "__main__":
    # Test installation
    install_packages()
    
    # Test model download
    download_model_if_needed()
    
    # Test RAG processing
    test_text = """
    This is a sample transcript.
    It contains multiple sentences.
    We will use this to test the RAG processor.
    The RAG processor should chunk this text and create embeddings.
    Then we can retrieve relevant chunks for a query.
    """
    
    rag = RAGProcessor()
    rag.process_transcript(test_text)
    
    # Test retrieval
    chunks = rag.retrieve_relevant_chunks("What does the RAG processor do?")
    print("Retrieved chunks:", chunks)