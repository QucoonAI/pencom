"""
Simplified RAG System - 4 Core Functions
========================================
1. Process complete document (PDF → Chunks → Vectors → Pinecone)
2. Add document to existing collection
3. Replace entire database with new document
4. Ask questions with RAG retrieval

Perfect for backend developers - clean, simple API
"""

import uuid
import time
import json 
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import boto3
import tiktoken
from io import BytesIO
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec

from config import settings
from src.utils import load_llm_prompt, render_prompt


# Use the logger initialized in the main app module or configure one here
logger = logging.getLogger(__name__)

# Load environment variables from .env file
aws_region =  settings.AWS_REGION# Define AWS region
S3_BUCKET = settings.S3_BUCKET_NAME  # Get S3 bucket from env
s3_client = boto3.client('s3', region_name=aws_region)  # Initialize Boto3 S3 client

class SimplifiedRAG:
    """Simplified RAG system with 4 core functions for backend integration"""
        
    def __init__(self):
        """Initialize the RAG system with AWS Bedrock and Pinecone"""
        try:
            # AWS Bedrock setup
            self.bedrock = boto3.client(
                'bedrock-runtime',
                region_name='us-east-1',
                # aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                # aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # Pinecone setup
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            
            # Create index if it doesn't exist
            if self.index_name not in [index.name for index in pc.list_indexes()]:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # Titan v2 embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            # Connect to the Pinecone index
            self.index = pc.Index(self.index_name)
            
            # Model configurations
            self.embedding_model = "amazon.titan-embed-text-v2:0"  # Embedding model
            self.chat_model = "us.amazon.nova-pro-v1:0"  # AWS Nova Pro LLM
            
            # Tokenizer for chunking (matches Claude models)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            logger.info("Simplified RAG system initialized successfully!") # Changed from print
        except Exception as e:
            logger.error(f"Failed to initialize SimplifiedRAG: {e}", exc_info=True)
            raise  # Re-raise exception to stop application startup if init fails
    
    def _extract_pdf_text(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text from PDF with page metadata"""
        try:
            # Create a PdfReader object from the in-memory file bytes
            reader = PdfReader(BytesIO(file_bytes))
            pages = []
            
            # Iterate through each page in the PDF
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text().strip()  # Extract and clean text
                
                # Only add pages that contain text
                if text:
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'char_count': len(text)
                    })
            
            logger.info(f"Extracted {len(pages)} pages with text from PDF.")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}", exc_info=True)
            # Propagate the error to be caught by the calling function
            raise Exception(f"Failed to extract PDF text: {str(e)}")


    def _get_s3_file_content(self, response, S3_BUCKET: str) -> bytes | None:
        """Retrieve the first PDF file from S3 and return its bytes."""
        try:
            # Loop through the files listed in the S3 response
            for obj in response.get("Contents", []):
                key = obj["Key"]
                # Find the first file that ends with .pdf
                if key.lower().endswith(".pdf"):
                    logger.info(f"Fetching PDF from S3: s3://{S3_BUCKET}/{key}")
                    # Get the file object from S3
                    file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                    # Read the file's content into bytes and return it
                    return file_obj["Body"].read()  # return bytes immediately

            # If no PDF file is found
            logger.warning("No PDF files found in S3 response.")
            return None

        except Exception as e:
            # Log any error during S3 retrieval
            logger.error(f"Error retrieving PDF from S3: {e}", exc_info=True)
            return None
       

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using AWS Bedrock Titan"""
        embeddings = []
        
        # Iterate over each text chunk
        for text in texts:
            try:
                # Format the request body for Bedrock Titan
                request_body = json.dumps({"inputText": text})
                
                # Invoke the Bedrock model
                response = self.bedrock.invoke_model(
                    modelId=self.embedding_model,
                    body=request_body,
                    contentType='application/json'
                )
                
                # Parse the response
                result = json.loads(response['body'].read())
                # Append the resulting embedding vector
                embeddings.append(result['embedding'])
                
            except Exception as e:
                # Log a warning and append a zero vector as a fallback
                logger.warning(f"Failed to generate embedding for text chunk: {str(e)}")
                # Use zero vector as fallback to avoid dimension mismatch
                embeddings.append([0.0] * 1024)
        
        logger.info(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    
    def _upload_to_pinecone(self, chunks: List[Dict], embeddings: List[List[float]], 
                            document_id: str, filename: str) -> Dict[str, Any]:
        """Upload chunks and embeddings to Pinecone with rich metadata"""
        try:
            vectors = []
            timestamp = datetime.now().isoformat()  # Get current timestamp
            
            # Prepare vectors for upload
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{i}"  # Create a unique ID for each chunk
                
                # Create a rich metadata object
                metadata = {
                    'document_id': document_id,
                    'filename': filename,
                    'page_number': chunk['page_number'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],  # Full Q&A text
                    'question': chunk.get('question', ''),  # Extracted question
                    'answer': chunk.get('answer', ''),  # Extracted answer
                    'token_count': chunk['token_count'],
                    'char_count': chunk['char_count'],
                    'created_at': timestamp,
                    'chunk_type': 'qa'  # Changed from 'text' to 'qa'
                }
                
                # Append the final vector object
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upload in batches for efficiency and reliability
            batch_size = 100
            total_uploaded = 0
            
            logger.info(f"Uploading {len(vectors)} vectors in batches of {batch_size}...")
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)  # Upsert batch to Pinecone
                total_uploaded += len(batch)
            
            logger.info(f"Successfully uploaded {total_uploaded} vectors to Pinecone.")
            
            # Return a summary of the upload
            return {
                'vectors_uploaded': total_uploaded,
                'document_id': document_id,
                'timestamp': timestamp
            }
        except Exception as e:
            logger.error(f"Failed to upload vectors to Pinecone: {e}", exc_info=True)
            raise Exception(f"Pinecone upload failed: {str(e)}") # Propagate error
    
    # =================
    # CORE FUNCTIONS
    # =================
    def _create_chunks(self, pages: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse document into Q&A pairs as chunks.
        Each chunk contains one complete question and its answer.
        """
        try:
            chunks = []
            chunk_index = 0
            
            # Combine all pages into one text for parsing
            full_text = ""
            page_boundaries = []  # Track where each page starts
            
            for page in pages:
                page_boundaries.append({
                    'page_number': page['page_number'],
                    'start_char': len(full_text)
                })
                full_text += page['text'] + "\n\n"
            
            # Split text into lines for parsing
            lines = full_text.split('\n')
            
            current_question = None
            current_answer = []
            current_page = 1
            char_position = 0
            
            for line in lines:
                line_stripped = line.strip()
                
                # Update current page based on character position
                for pb in page_boundaries:
                    if char_position >= pb['start_char']:
                        current_page = pb['page_number']
                
                # Check if line starts with "Q:" - new question
                if line_stripped.startswith('Q:'):
                    # If we have a previous Q&A pair, save it as a chunk
                    if current_question and current_answer:
                        answer_text = ' '.join(current_answer).strip()
                        qa_text = f"{current_question}\n{answer_text}"
                        
                        chunks.append({
                            'text': qa_text,
                            'page_number': current_page,
                            'token_count': len(self.tokenizer.encode(qa_text)),
                            'char_count': len(qa_text),
                            'chunk_index': chunk_index,
                            'question': current_question.replace('Q:', '').strip(),
                            'answer': answer_text.replace('A:', '').strip()
                        })
                        chunk_index += 1
                    
                    # Start new question
                    current_question = line_stripped
                    current_answer = []
                
                # Check if line starts with "A:" - answer to current question
                elif line_stripped.startswith('A:'):
                    current_answer = [line_stripped]
                
                # Continue building the answer (multi-line answers)
                elif current_answer and line_stripped:
                    # Skip section headers (e.g., "Section 1:", "Section 2:")
                    if not line_stripped.lower().startswith('section'):
                        current_answer.append(line_stripped)
                
                # Update character position
                char_position += len(line) + 1  # +1 for newline
            
            # Don't forget the last Q&A pair
            if current_question and current_answer:
                answer_text = ' '.join(current_answer).strip()
                qa_text = f"{current_question}\n{answer_text}"
                
                chunks.append({
                    'text': qa_text,
                    'page_number': current_page,
                    'token_count': len(self.tokenizer.encode(qa_text)),
                    'char_count': len(qa_text),
                    'chunk_index': chunk_index,
                    'question': current_question.replace('Q:', '').strip(),
                    'answer': answer_text.replace('A:', '').strip()
                })
            
            logger.info(f"Created {len(chunks)} Q&A chunks from {len(pages)} pages.")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed during Q&A chunking: {e}", exc_info=True)
            raise Exception(f"Q&A chunking failed: {str(e)}")


    def _process_complete_document(self, filename: str) -> Dict[str, Any]:
        """
        Complete PDF Processing Pipeline
        
        Takes a PDF, processes it completely: PDF → Chunks → Embeddings → Pinecone
        Perfect for backend developers - one call does everything.
        
        Args:
            filename: Name of the PDF file in S3
            
        Returns:
            Dict with processing results and metadata for backend tracking
        """
        start_time = time.time()
        
        try:
            # Generate document ID and name
            document_id = str(uuid.uuid4())  # Unique ID for this document
            
            logger.info(f"Processing document: {filename} (ID: {document_id})")
            
            # Step 1: Extract PDF text
            logger.info("Extracting file bytes from S3...")
            # Find the file in S3
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=f"{filename.lower().replace(' ', '_')}.pdf"
            )
            
            # Handle file not found
            if 'Contents' not in response:
                logger.error(f"File not found in S3 for: {filename}")
            
            # Get file content and extract text
            file_bytes = self._get_s3_file_content(response, S3_BUCKET)
            pages = self._extract_pdf_text(file_bytes) #type: ignore
            total_pages = len(pages)
            
            # Step 2: Create chunks
            logger.info("Creating chunks...")
            chunks = self._create_chunks(pages)
            total_chunks = len(chunks)
            if total_chunks == 0:
                raise Exception("No text chunks were created from the PDF.")
            
            # Step 3: Generate embeddings
            logger.info("Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self._generate_embeddings(chunk_texts)
            
            # Step 4: Upload to Pinecone
            logger.info("Uploading to Pinecone...")
            upload_result = self._upload_to_pinecone(chunks, embeddings, document_id, filename)
            
            # Calculate processing time and statistics
            processing_time = time.time() - start_time
            avg_chunk_length = sum(chunk['char_count'] for chunk in chunks) / len(chunks)
            total_tokens = sum(chunk['token_count'] for chunk in chunks)
            
            # Prepare success response
            result = {
                'success': True,
                'document_id': document_id,
                'filename': filename,
                'processing_time_seconds': round(processing_time, 2),
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'total_tokens': total_tokens,
                'chunking_strategy': 'qa_pairs',  # Q&A based chunking
                'avg_chunk_length': round(avg_chunk_length, 1),
                'pinecone_vectors_uploaded': upload_result['vectors_uploaded'],
                'created_at': upload_result['timestamp'],
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'index_name': self.index_name
                }
            }
            
            logger.info(f"✅ SUCCESS! Document processed completely in {processing_time:.2f}s") # Changed from print
            return result
            
        except Exception as e:
            logger.error(f"❌ FAILED to process document {filename}: {e}", exc_info=True)
            # Return error response
            return {
                'success': False,
                'error': str(e),
                'document_id': None,
                'processing_time_seconds': round(time.time() - start_time, 2)
            }


    def add_to_existing_collection(self, filename: str) -> Dict[str, Any]:
        """
        Add Document to Existing Collection
        
        Adds a new document to the existing Pinecone database without removing anything.
        Perfect for expanding your knowledge base.
        
        Args:
            filename: Name of the document
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Adding document '{filename}' to existing collection...")
        
        try:
            # Get current document count with retry
            initial_vector_count = 0
            try:
                stats = self.index.describe_index_stats()
                initial_vector_count = stats['total_vector_count']
                logger.info(f"Collection has {initial_vector_count} vectors before adding.")
            except Exception as e:
                logger.warning(f"Could not get initial stats (will skip): {e}")
            
            # Process the document (same as function 1)
            result = self._process_complete_document(filename)
            
            # If processing was successful, add collection info to the result
            if result['success']:
                # Get new stats after adding (with retry and fallback)
                try:
                    new_stats = self.index.describe_index_stats()
                    result['collection_info'] = {
                        'total_vectors_before': initial_vector_count,
                        'total_vectors_after': new_stats['total_vector_count'],
                        'vectors_added': result.get('pinecone_vectors_uploaded', 0)
                    }
                    logger.info(f"✅ Document added! Collection now has {new_stats['total_vector_count']} total vectors")
                except Exception as e:
                    logger.warning(f"Could not get final stats: {e}")
                    result['collection_info'] = {
                        'total_vectors_before': initial_vector_count,
                        'vectors_added': result.get('pinecone_vectors_uploaded', 0),
                        'note': 'Final count unavailable due to connection timeout'
                    }
                    logger.info(f"✅ Document added! {result.get('pinecone_vectors_uploaded', 0)} vectors uploaded")
            else:
                logger.error(f"Failed to add document '{filename}': {result.get('error')}")

            return result
        
        except Exception as e:
            logger.error(f"❌ FAILED to add document {filename} to collection: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


    def replace_specific_document_vectors(self, filename: str) -> Dict[str, Any]:
        """
        Replace all vectors in Pinecone associated with a specific document.
        
        Only deletes vectors with matching 'filename' metadata, then uploads new vectors.
        
        Args:
            filename: Name of the document
            document_id: Unique ID of the document (used in vector IDs)
        
        Returns:
            Dict with processing results
        """
        logger.info(f"Replacing vectors for document: {filename}")
        
        try:
            # Get count of vectors with this filename
            self.index.delete(filter={"filename": {"$eq": filename}})

            logger.warning(f"Deleting vectors associated with {filename}...")
            
            # Delete all vectors with metadata filter
            self.index.delete(filter={"filename": {"$eq": filename}})
            logger.info(f"Deleted  vectors for {filename}.")
            
            # Process new document and get chunks + embeddings
            result = self._process_complete_document(filename)
            
            # Add replacement info
            if result.get('success'):
                result['document_replacement_info'] = {
                    'new_vectors_uploaded': result.get('pinecone_vectors_uploaded', 0),
                    'replacement_completed': True
                }
                
                logger.info(f"Replacement complete")
            else:
                logger.error(f"Replacement failed for {filename}: {result.get('error')}")
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to replace vectors for {filename}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'document_replacement_info': {
                    'new_vectors_uploaded': 0,
                    'replacement_completed': False
                }
            }


    def reset_vector_database(self) -> Dict[str, Any]:
        """
        Empty Entire Database
        
        Deletes ALL existing documents.
        Use with caution - this wipes everything!
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Deleting entire database")
        
        try:
            # Get current stats before deleting
            initial_stats = self.index.describe_index_stats()
            initial_count = initial_stats['total_vector_count']
            
            logger.warning(f"Deleting {initial_count} existing vectors...")
            
            # Delete all existing vectors
            self.index.delete(delete_all=True)
            
            logger.info("Database cleared!")
                       
            return {
                'success':True,
                'vectors_deleted': initial_count,
                'reset_completed': True
            }
            
        except Exception as e:
            logger.error(f"FAILED to rest database: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'database_replacement_info': {
                    'vectors_deleted': initial_count, # Report how many were deleted before fail
                    'new_vectors_uploaded': 0,
                    'replacement_completed': False
                }
            }
    

    def ask_questions(self, question: str) -> Dict[str, Any]:
        """
        Ask Questions with RAG Retrieval
        
        Query the knowledge base and get AI-generated answers with sources.
        Uses static top_k=5 for consistent retrieval.
        
        Args:
            question: The question to ask
            
        Returns:
            Dict with answer, sources, and metadata
        """
        top_k = 5  # Static value for consistent retrieval
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question[:100]}...") # Log truncated question
            
            # Step 1: Generate question embedding
            question_embedding = self._generate_embeddings([question])[0]
            
            # Step 2: Search Pinecone for relevant chunks
            logger.info(f"Querying Pinecone with top_k={top_k}...")
            search_results = self.index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True  # Get metadata for sources
            )
            
            # Handle no results found
            # if not search_results['matches']:
            #     logger.warning(f"No relevant documents found for question: {question[:50]}...")
            #     return {
            #         'success': False,
            #         'error': 'No relevant documents found in the knowledge base',
            #         'answer': "I'm sorry, I couldn't find any relevant information in the knowledge base to answer that question.",
            #         'sources': [],
            #         'query_time_seconds': round(time.time() - start_time, 2)
            #     }
            
            # Step 3: Prepare context and sources for Claude
            context_chunks = []
            sources = []
            
            logger.info(f"Retrieved {len(search_results['matches'])} context chunks.")
            
            for match in search_results['matches']:
                metadata = match['metadata']
                # Add the actual text to the context
                context_chunks.append(metadata['text'])
                # Add source info for citation
                sources.append({
                    'document_id': metadata['document_id'],
                    'filename': metadata['filename'],
                    'page_number': metadata['page_number'],
                    'relevance_score': round(match['score'], 3),
                    'chunk_index': metadata['chunk_index']
                })
            
            # Step 4: Generate answer with Nova Pro
            context_text = "\n\n".join(context_chunks)
            
            # Create the prompt with context using imported prompt template
            system_prompt, user_prompt_template = load_llm_prompt(
                "prompt/rag_system_prompt.yaml"
            )

            prompt = render_prompt(
                    user_prompt_template,
                    context_text=context_text,
                    question=question
                )
            print(prompt)
            # Format the request for AWS Nova Pro
            request_body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": system_prompt}],
                "inferenceConfig": {
                    "max_new_tokens": 2000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            })
            
            logger.info("Generating answer with AWS Nova Pro...")
            # Invoke the Bedrock model
            response = self.bedrock.invoke_model(
                modelId=self.chat_model,
                body=request_body,
                contentType='application/json'
            )
            print(response)
            # Parse the response from Nova Pro
            result = json.loads(response['body'].read())
            # Extract the text answer (Nova Pro format)
            if 'output' in result and 'message' in result['output']:
                answer = result['output']['message']['content'][0]['text']
            else:
                answer = 'No answer generated'
            
            query_time = time.time() - start_time
            logger.info(f"Successfully answered question in {query_time:.2f}s")
            
            # Return the complete response
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'question': question,
                'query_time_seconds': round(query_time, 2),
                'chunks_retrieved': len(context_chunks),
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'chat_model': self.chat_model,
                    'top_k_used': top_k
                }
            }
            
        except Exception as e:
            logger.error(f"FAILED to answer question '{question[:50]}...': {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'answer': None,
                'sources': [],
                'query_time_seconds': round(time.time() - start_time, 2)
            }
    

    # =================
    # UTILITY FUNCTIONS
    # =================
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        try:
            # Get stats directly from Pinecone index
            stats = self.index.describe_index_stats()
            logger.info(f"Retrieved DB stats: {stats}")
            return {
                'total_vectors': stats['total_vector_count'],
                'index_fullness': stats.get('index_fullness', 0), # Serverless may not have this
                'dimension': stats.get('dimension', 1024), # Get dimension if available
                'index_name': self.index_name
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}", exc_info=True)
            return {'error': str(e)}
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database with metadata"""
        try:
            logger.info("Listing all documents... (uses dummy query)")
            # Query with a dummy vector to get a sample of vectors
            # This is a workaround as Pinecone doesn't have a "list all" metadata API
            sample_results = self.index.query(
                vector=[0.0] * 1024,  # Dummy vector
                top_k=1000,  # Get many results to find all documents
                include_metadata=True
            )
            
            # Group by document_id to aggregate document info
            documents = {}
            for match in sample_results['matches']:
                metadata = match['metadata']
                doc_id = metadata.get('document_id', 'unknown')
                
                # If this is the first time seeing this doc_id, initialize it
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': metadata.get('filename', 'unknown'),
                        'created_at': metadata.get('created_at', 'unknown'),
                        'chunk_count': 0
                    }
                # Increment the chunk count for this document
                documents[doc_id]['chunk_count'] += 1
            
            logger.info(f"Found {len(documents)} unique documents.")
            # Return the aggregated list of documents
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to list all documents: {e}", exc_info=True)
            return [] # Return empty list on failure