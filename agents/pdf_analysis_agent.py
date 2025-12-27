"""
PDF Analysis Agent

Specialized agent for processing and analyzing PDF documents
from legal sources with error capture for self-healing.
Uses PyMuPDF (fitz) for robust text extraction.
"""

import asyncio
import os
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF analysis will be simulated.")
    print("Install with: pip install PyMuPDF")


class AutonomousPdfAnalysisAgent:
    """Agent for analyzing PDF documents from legal sources."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("pdf_analysis_agent")
        self.name = "pdf_analysis_agent"
        self.supported_formats = config.get('supported_formats', ['pdf'])
        self.output_dir = Path(config.get('output_dir', 'data/production/pdfs/processed'))
        self.raw_dir = Path(config.get('raw_files', 'data/production/pdfs/raw'))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Error tracking for self-healing
        self.error_count = 0
        self.last_error = None
        
        self.logger.info(f"PDF Analysis Agent initialized (PyMuPDF: {PYMUPDF_AVAILABLE})")
    
    async def initialize(self) -> bool:
        """Initialize the PDF analysis agent"""
        try:
            self.logger.info("Initializing PDF analysis components...")
            
            if not PYMUPDF_AVAILABLE:
                self.logger.warning("PyMuPDF not available, using mock PDF analysis")
                return True
            
            # Test PyMuPDF functionality
            try:
                # Create a test document to verify functionality
                test_doc = fitz.open()  # Create empty PDF
                test_doc.close()
                self.logger.info("PyMuPDF functionality verified")
            except Exception as e:
                self.logger.error(f"PyMuPDF test failed: {e}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PDF analysis agent: {e}")
            return False
    
    async def scan_raw_directory(self) -> List[str]:
        """
        Scan the data/production/pdfs/raw/ directory for PDF files
        
        Returns:
            List of PDF file paths found
        """
        try:
            self.logger.info(f"Scanning raw PDF directory: {self.raw_dir}")
            
            if not self.raw_dir.exists():
                self.logger.warning(f"Raw directory does not exist: {self.raw_dir}")
                return []
            
            # Find all PDF files
            pdf_files = list(self.raw_dir.glob("*.pdf"))
            pdf_paths = [str(f) for f in pdf_files]
            
            self.logger.info(f"Found {len(pdf_paths)} PDF files in raw directory")
            return pdf_paths
            
        except Exception as e:
            self.logger.error(f"Failed to scan raw directory: {e}")
            await self._handle_error(e, "scan_raw_directory", {})
            return []
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using PyMuPDF (fitz)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            self.logger.info(f"Extracting text from PDF: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Generate document ID
            doc_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
            
            if not PYMUPDF_AVAILABLE:
                # Mock extraction for development
                return await self._mock_extraction(pdf_path, doc_id)
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = doc.metadata
            page_count = doc.page_count
            
            # Extract text from all pages
            full_text = ""
            pages_text = []
            
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
                full_text += page_text + "\n"
            
            doc.close()
            
            # Clean and structure the extracted text
            cleaned_text = self._clean_extracted_text(full_text)
            
            # Save processed text to output directory
            output_file = self.output_dir / f"{doc_id}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Create analysis result
            analysis_result = {
                'status': 'success',
                'doc_id': doc_id,
                'file_path': pdf_path,
                'file_name': os.path.basename(pdf_path),
                'output_file': str(output_file),
                'extraction_timestamp': datetime.now().isoformat(),
                'text_extracted': True,
                'metadata': {
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'subject': metadata.get('subject', ''),
                    'creator': metadata.get('creator', ''),
                    'producer': metadata.get('producer', ''),
                    'creation_date': metadata.get('creationDate', ''),
                    'modification_date': metadata.get('modDate', '')
                },
                'statistics': {
                    'pages_count': page_count,
                    'total_characters': len(full_text),
                    'total_words': len(full_text.split()),
                    'total_lines': len(full_text.split('\n')),
                    'file_size_bytes': os.path.getsize(pdf_path)
                },
                'pages_detail': pages_text,
                'text_preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
                'processing_time_seconds': 0.0,  # Will be set by caller
                'confidence_score': self._calculate_confidence_score(cleaned_text, page_count),
                'document_type': self._detect_document_type(cleaned_text, metadata),
                'keywords_found': self._extract_keywords(cleaned_text),
                'language_detected': self._detect_language(cleaned_text)
            }
            
            self.logger.info(f"Successfully extracted text from {os.path.basename(pdf_path)} "
                           f"({page_count} pages, {len(cleaned_text)} chars)")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            error_context = {
                'pdf_path': pdf_path,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            await self._handle_error(e, "extract_text_from_pdf", error_context)
            
            return {
                'status': 'error',
                'file_path': pdf_path,
                'error': str(e),
                'extraction_timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__
            }
    
    async def _mock_extraction(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:
        """Mock PDF extraction for development when PyMuPDF is not available"""
        self.logger.info(f"Performing mock extraction for: {pdf_path}")
        
        # Create mock text content
        mock_text = f"""
Mock extracted text from {os.path.basename(pdf_path)}

This is simulated PDF content for development purposes.
In a real implementation, this would contain the actual extracted text
from the PDF document using PyMuPDF.

Document ID: {doc_id}
File: {os.path.basename(pdf_path)}
Date: {datetime.now().isoformat()}

Sample legal content:
- Luật số 123/2024/QH15
- Nghị định 45/2024/NĐ-CP  
- Thông tư 12/2024/TT-BKHĐT

This document contains legal provisions and regulations
related to the implementation of government policies.
"""
        
        # Save mock text to output directory
        output_file = self.output_dir / f"{doc_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mock_text)
        
        return {
            'status': 'success',
            'doc_id': doc_id,
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'output_file': str(output_file),
            'extraction_timestamp': datetime.now().isoformat(),
            'text_extracted': True,
            'metadata': {
                'title': f'Mock Title for {os.path.basename(pdf_path)}',
                'author': 'Mock Author',
                'subject': 'Legal Document',
                'creator': 'Mock Creator',
                'producer': 'Mock Producer',
                'creation_date': '',
                'modification_date': ''
            },
            'statistics': {
                'pages_count': 10,
                'total_characters': len(mock_text),
                'total_words': len(mock_text.split()),
                'total_lines': len(mock_text.split('\n')),
                'file_size_bytes': os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
            },
            'pages_detail': [
                {
                    'page_number': i + 1,
                    'text': f"Mock text for page {i + 1}",
                    'char_count': 25
                }
                for i in range(10)
            ],
            'text_preview': mock_text[:500] + "..." if len(mock_text) > 500 else mock_text,
            'processing_time_seconds': 2.5,
            'confidence_score': 0.85,
            'document_type': 'legal_document',
            'keywords_found': ['luật', 'nghị định', 'chính sách', 'quy định', 'mock'],
            'language_detected': 'vietnamese'
        }
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        artifacts = [
            '\x0c',  # Form feed
            '\x00',  # Null bytes
            'Â',     # Common encoding artifact
            '€',     # Euro symbol artifact
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive consecutive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def _calculate_confidence_score(self, text: str, page_count: int) -> float:
        """Calculate confidence score for the extraction"""
        if not text:
            return 0.0
        
        # Base score
        score = 0.5
        
        # Add points for content length
        if len(text) > 1000:
            score += 0.2
        elif len(text) > 500:
            score += 0.1
        
        # Add points for meaningful content
        word_count = len(text.split())
        if word_count > 100:
            score += 0.2
        elif word_count > 50:
            score += 0.1
        
        # Add points for multiple pages
        if page_count > 1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _detect_document_type(self, text: str, metadata: dict) -> str:
        """Detect document type based on content and metadata"""
        text_lower = text.lower()
        
        # Legal document indicators
        legal_keywords = ['luật', 'nghị định', 'thông tư', 'quyết định', 'vbpl', 'văn bản']
        if any(keyword in text_lower for keyword in legal_keywords):
            return 'legal_document'
        
        # Academic indicators
        academic_keywords = ['abstract', 'introduction', 'conclusion', 'references', 'doi:']
        if any(keyword in text_lower for keyword in academic_keywords):
            return 'academic_paper'
        
        # Business indicators
        business_keywords = ['invoice', 'receipt', 'contract', 'agreement', 'proposal']
        if any(keyword in text_lower for keyword in business_keywords):
            return 'business_document'
        
        return 'general_document'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the text"""
        if not text:
            return []
        
        # Common legal and business keywords
        legal_keywords = [
            'luật', 'nghị định', 'thông tư', 'quyết định', 'chính sách',
            'quy định', 'văn bản', 'pháp luật', 'hành chính', 'kinh tế',
            'đầu tư', 'thương mại', 'doanh nghiệp', 'công ty', 'hợp đồng'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in legal_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:10]  # Limit to top 10 keywords
    
    def _detect_language(self, text: str) -> str:
        """Detect document language based on content"""
        if not text:
            return 'unknown'
        
        # Simple Vietnamese detection
        vietnamese_chars = 'ăâêôơưđăâêôơưĐĂÂÊÔƠƯĐ'
        vietnamese_count = sum(1 for char in text if char in vietnamese_chars)
        
        if vietnamese_count > len(text) * 0.01:  # If more than 1% Vietnamese chars
            return 'vietnamese'
        
        # Simple English detection
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        english_count = sum(1 for word in english_words if word in text.lower().split())
        
        if english_count > 5:
            return 'english'
        
        return 'unknown'
    
    async def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF document and extract relevant information."""
        return await self.extract_text_from_pdf(pdf_path)
    
    async def batch_analyze(self, pdf_directory: str = None) -> List[Dict[str, Any]]:
        """Analyze multiple PDF files in a directory"""
        try:
            if pdf_directory:
                # Use provided directory
                scan_dir = Path(pdf_directory)
                self.logger.info(f"Starting batch analysis of directory: {pdf_directory}")
            else:
                # Use default raw directory
                scan_dir = self.raw_dir
                self.logger.info(f"Starting batch analysis of default raw directory: {scan_dir}")
            
            if not scan_dir.exists():
                raise FileNotFoundError(f"Directory not found: {scan_dir}")
            
            # Find all PDF files
            pdf_files = list(scan_dir.glob("*.pdf"))
            
            if not pdf_files:
                self.logger.info("No PDF files found in directory")
                return []
            
            results = []
            for pdf_file in pdf_files:
                start_time = datetime.now()
                result = await self.extract_text_from_pdf(str(pdf_file))
                
                # Add processing time
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                if 'statistics' in result:
                    result['statistics']['processing_time_seconds'] = processing_time
                
                results.append(result)
                
                # Small delay between files to avoid overwhelming system
                await asyncio.sleep(0.1)
            
            successful = sum(1 for r in results if r.get('status') == 'success')
            self.logger.info(f"Batch analysis completed: {successful}/{len(results)} PDF files processed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            await self._handle_error(e, "batch_analyze", {'pdf_directory': pdf_directory})
            return []
    
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.supported_formats
    
    async def _handle_error(self, error: Exception, function_name: str, context: dict):
        """Handle errors with context capture for self-healing"""
        self.error_count += 1
        self.last_error = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'function_name': function_name,
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        # Log the error
        self.logger.error(f"Error in {function_name}: {error}")
        
        # Send error to healing agent if available
        try:
            from utils.agent_registry import AgentRegistry
            registry = AgentRegistry()
            healing_agent = await registry.get_agent("healing_agent")
            
            if healing_agent:
                error_context = {
                    'timestamp': datetime.now().isoformat(),
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'agent_name': self.name,
                    'function_name': function_name,
                    'file_path': __file__,
                    'traceback': traceback.format_exc(),
                    'additional_context': context,
                    'severity': 'medium'
                }
                
                await healing_agent.process_message({
                    "type": "error_report",
                    "error_context": error_context
                })
                
                self.logger.info("Error reported to healing agent")
        except Exception as e:
            self.logger.warning(f"Failed to report error to healing agent: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_name': self.name,
            'status': 'running',
            'error_count': self.error_count,
            'last_error': self.last_error,
            'pymupdf_available': PYMUPDF_AVAILABLE,
            'raw_directory': str(self.raw_dir),
            'output_directory': str(self.output_dir),
            'supported_formats': self.supported_formats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("PDF Analysis Agent cleanup completed")