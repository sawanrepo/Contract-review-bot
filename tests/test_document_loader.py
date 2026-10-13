import unittest
import os
import tempfile
import sys

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from document_loader import load_documents
from langchain_core.documents import Document

class DummyUploadFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return memoryview(self._content)

class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_txt_file(self):
        txt_path = os.path.join(self.temp_dir.name, "sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Section 1: The parties agree to confidential terms.\n\nSection 2: Governing law is California.")

        chunks = load_documents(txt_path)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Document)
        self.assertEqual(chunks[0].metadata.get("page_number"), 1)
        self.assertIn("confidential", chunks[0].page_content)

    def test_chunking_behavior(self):
        long_content = "Clause A. " * 300  # ~3000 chars
        txt_path = os.path.join(self.temp_dir.name, "long_sample.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(long_content)

        chunks = load_documents(txt_path)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Chunk size limit is 900
            self.assertLessEqual(len(chunk.page_content), 1000)

    def test_unsupported_file_type(self):
        csv_path = os.path.join(self.temp_dir.name, "data.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("col1,col2\nval1,val2")

        with self.assertRaises(ValueError) as ctx:
            load_documents(csv_path)
        self.assertIn("Unsupported file type", str(ctx.exception))

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError) as ctx:
            load_documents(12345)
        self.assertIn("Invalid input", str(ctx.exception))

    def test_uploaded_file_cleanup(self):
        upload = DummyUploadFile("dummy_contract.txt", b"Payment shall be made in USD within 30 days.")
        chunks = load_documents(upload)
        self.assertGreater(len(chunks), 0)
        # Verify temporary file was cleaned up and does not linger in temp/
        temp_folder = "temp"
        if os.path.exists(temp_folder):
            remaining = [f for f in os.listdir(temp_folder) if f.startswith(f"upload_{os.getpid()}_dummy_contract.txt")]
            self.assertEqual(len(remaining), 0, "Temporary upload file was not cleaned up!")

if __name__ == "__main__":
    unittest.main()
