import os
from dotenv import load_dotenv
from tests.test_advanced_rag import AdvancedRAGTester

def main():
    load_dotenv()
    
    print("🚀 ADVANCED RAG SYSTEM - COMPLETE IMPLEMENTATION")
    print("=" * 60)
    
    # Initialize and run tests
    tester = AdvancedRAGTester()
    
    # You need to upload a PDF file first in Colab
    pdf_path = "your_document.pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        results = tester.test_complete_pipeline(pdf_path)
    else:
        print("❌ PDF file not found. Please upload your PDF file.")
        print("📁 Expected file:", pdf_path)

if __name__ == "__main__":
    main()