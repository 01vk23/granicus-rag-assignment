from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber


class Document:
    def __init__(self, doc_id: str, source: str, doc_type: str, content: str):
        self.doc_id = doc_id
        self.source = source
        self.doc_type = doc_type
        self.content = content


class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def discover_files(self) -> List[Path]:
        return [p for p in self.data_dir.iterdir() if p.is_file()]

    
    def load(self) -> List[Document]:
        documents = []

        import uuid

        for path in self.discover_files():
            file_type = self.detect_file_type(path)

            if file_type == "binary" or file_type == "unknown":
                continue

            if file_type == "text":
                content = self.read_text(path)

            elif file_type == "csv":
                content = self.read_csv(path)

            elif file_type == "pdf":
                content = self.read_pdf(path)

            elif file_type == "html":
                content = self.read_text(path)

            else:
                continue

            content = content.strip()

            if not content or len(content) < 50:
                continue

            documents.append(
                Document(
                    doc_id=str(uuid.uuid4()),
                    source=path.name,
                    doc_type=file_type,
                    content=content
                )
            )

        return documents

    

    def detect_file_type(self, path: Path) -> str:
        """
        Detect actual file type using content-based sniffing,
        but trust .csv extension explicitly.
        """
        suffix = path.suffix.lower()

        # Explicit extension checks first
        if suffix == ".csv":
            return "csv"

        if suffix in [".txt", ".md"]:
            return "text"

        try:
            with open(path, "rb") as f:
                header = f.read(1024)
        except Exception:
            return "unknown"

        # Real PDF
        if header.startswith(b"%PDF"):
            return "pdf"

        # HTML disguised as PDF or text
        if b"<html" in header.lower():
            return "html"

        # Fallback to text if decodable
        try:
            header.decode("utf-8")
            return "text"
        except UnicodeDecodeError:
            return "binary"

    def read_text(self, path: Path) -> str:
        """
        Read plain text / markdown safely.
        """
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        
    def read_csv(self, path: Path) -> str:
        """
        Read CSV and convert each row into semantic text.
        This improves retrieval grounding.
        """
        import csv

        rows = []

        try:
            with open(path, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = next(reader, None)

                for row in reader:
                    if headers:
                        parts = [
                            f"{h.strip()}: {v.strip()}"
                            for h, v in zip(headers, row)
                            if v.strip()
                        ]
                        rows.append(" | ".join(parts))
                    else:
                        rows.append(" | ".join(cell.strip() for cell in row))

        except Exception:
            return ""

        return "\n".join(rows)


    def read_pdf(self, path: Path) -> str:
        """
        Extract text and tables from PDF using pdfplumber.
        Tables are flattened into row-wise semantic text.
        """
        

        text_blocks = []
        table_blocks = []

        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_blocks.append(page_text)

                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            row_text = " | ".join(cell.strip() for cell in row if cell)
                            if row_text:
                                table_blocks.append(row_text)

        except Exception:
            return ""

        content = "\n".join(text_blocks)

        if table_blocks:
            content += "\n\n=== EXTRACTED TABLE DATA ===\n"
            content += "\n".join(table_blocks)

        return content



