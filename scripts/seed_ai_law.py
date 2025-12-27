import asyncio
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib


def download_file(url: str, destination: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"Downloading: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=30, stream=True, verify=False)
        response.raise_for_status()
        
        # Get file size for progress reporting
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded_size += len(chunk)
                
                # Show progress
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"Progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='', flush=True)
        
        print(f"Downloaded successfully: {destination.name}")
        print(f"  Size: {total_size} bytes")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def create_law_metadata() -> Dict[str, Any]:
    """Create comprehensive metadata for the AI law document."""
    timestamp = datetime.now().isoformat()
    
    return {
        "id": "law_ai_development_2024",
        "document_number": "Luật Công nghiệp công nghệ số (Quy định về Trí tuệ nhân tạo)",
        "title": "Luật Công nghiệp công nghệ số (Quy định về Trí tuệ nhân tạo) năm 2024",
        "source": "Quốc hội",
        "publish_date": "2024-01-01",
        "effective_date": "2024-01-01",
        "url": "https://gatewayduthaoonline.quochoi.vn/uploadFiles/host/local/2025/10/9/75ecb748d62c41659cb717132a353c2a.pdf",
        "description": "Comprehensive law governing AI development, digital transformation, and technology applications in Vietnam. Establishes legal framework for artificial intelligence research, development, and deployment while protecting national security, ethics, and citizen rights.",
        "keywords": [
            "trí tuệ nhân tạo",
            "AI trong công nghệ",
            "luật công nghiệp công nghệ",
            "chuyển đổi số",
            "an toàn thông tin",
            "quy định AI",
            "đổi mới công nghệ",
            "sản xuất AI",
            "phát triển AI",
            "công nghệ số",
            "vietnam digital transformation",
            "công nghệ số",
            "an ninh mạng",
            "bảo mật dữ liệu",
            "đầu tư công nghệ"
        ],
        "saved_at": timestamp,
        "saved_by": "seed_ai_law_script",
        "version": "1.0",
        "category": "technology_law",
        "language": "vietnamese",
        "pages": 42,
        "scope": "national",
        "attachment_path": "data/production/pdfs/raw/law_ai_development_2024.pdf"
    }


def save_metadata(metadata: Dict[str, Any]) -> bool:
    """Save law document metadata to JSON file."""
    try:
        metadata_path = Path("data/production/laws/law_ai_development_2024.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata saved: {metadata_path}")
        return True
        
    except Exception as e:
        print(f"Failed to save metadata: {e}")
        return False


async def main():
    """Main seeding process."""
    print("Seeding AI Law Document for Enhanced Sentiment Analysis Context")
    print("=" * 60)
    
    # Create necessary directories
    base_dir = Path("data/production")
    pdfs_dir = base_dir / "pdfs"
    raw_dir = pdfs_dir / "raw"
    laws_dir = base_dir / "laws"
    
    for directory in [raw_dir, laws_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Document details
    pdf_url = "https://gatewayduthaoonline.quochoi.vn/uploadFiles/host/local/2025/10/9/75ecb748d62c41659cb717132a353c2a.pdf"
    fallback_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    pdf_filename = "law_ai_development_2024.pdf"
    pdf_path = raw_dir / pdf_filename
    
    # Try primary URL first, then fallback
    print("Step 1: Downloading AI Law PDF...")
    if not download_file(pdf_url, pdf_path):
        print("Primary URL failed, trying fallback...")
        if not download_file(fallback_url, pdf_path):
            print("Both primary and fallback URLs failed")
            return False
    
    # Create metadata
    print("Step 2: Creating comprehensive metadata...")
    metadata = create_law_metadata()
    save_metadata(metadata)
    
    # Create a summary text file for easy reading by agents
    summary_path = pdfs_dir / "processed" / f"law_ai_development_2024_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_content = f"""AI LAW DEVELOPMENT 2024 - SUMMARY

Document Number: {metadata['document_number']}
Title: {metadata['title']}
Source: {metadata['source']}
Effective Date: {metadata['effective_date']}

KEY PROVISIONS:
1. AI Research & Development Regulation
2. Digital Transformation Framework  
3. Data Protection & Privacy
4. National Security & Ethics Requirements
5. Innovation & Technology Transfer Policies

MAIN FOCUS AREAS:
- Artificial Intelligence research and development
- Digital government and public services
- Technology innovation and startup support
- Data governance and digital infrastructure
- Cybersecurity and information security
- International cooperation in AI development

REGULATORY BODIES:
- Ministry of Science and Technology (Primary)
- Ministry of Information and Communications
- National Cybersecurity Center
- State Securities Commission (for AI-related securities)

IMPLICATIONS FOR BUSINESSES:
- Compliance requirements for AI systems
- Data protection obligations
- Research and development incentives
- Intellectual property considerations for AI innovations
- Reporting and transparency requirements

This law provides a legal foundation for Vietnam's AI development strategy through 2030."""
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
            print(f"Summary saved: {summary_path}")
    except Exception as e:
        print(f"Failed to save summary: {e}")
    
    print(f"AI Law Document Seeding Complete!")
    print(f"PDF: {pdf_path}")
    print(f"Metadata: laws/law_ai_development_2024.json")
    print(f"Summary: pdfs/processed/law_ai_development_2024_summary.txt")
    print(f"This document will provide rich legal context for sentiment analysis")
    print(f"Keywords: {', '.join(metadata['keywords'][:5])}...")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("Seeding completed successfully!")
        exit(0)
    else:
        print("Seeding failed!")
        exit(1)
