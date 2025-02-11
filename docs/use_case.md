### Use Case: LLM-Driven OCR Workflow for EIN Extraction

1. **Extract EIN from PDF:**
   - Source: [Wells Fargo 10-K Report (2023)](https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2023/10k.pdf)
   - Task: Identify and extract the **Employer Identification Number (EIN)** from the PDF.

2. **Cross-Reference EIN in SEC Filings:**
   - Look for the extracted EIN in the following SEC filings:
     - [SEC Filing 1 (D399581D8K)](https://www.sec.gov/Archives/edgar/data/19617/000119312522236574/d399581d8k.htm)
     - [SEC Filing 2 (D477710D8K)](https://www.sec.gov/Archives/edgar/data/72971/000119312522221329/d477710d8k.htm)

3. **Key Extraction from URL:**
   - For each URL containing the EIN, extract:
     - **Date signed**
     - **Name and Title of the Officer**

