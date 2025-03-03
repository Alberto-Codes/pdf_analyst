# PDF Analyst Project State

## Project Overview

The PDF Analyst (pdf_analyst) is a Python-based tool designed to extract structured information from PDF documents, particularly focusing on SEC filings like 10-K reports. The project uses Optical Character Recognition (OCR) and LLM processing through Google's Gemini API to extract specific information from documents.

## Key Use Cases

The primary use case identified is LLM-driven OCR workflow for information extraction from financial documents:

1. Extract Employer Identification Numbers (EIN) from PDF documents (e.g., Wells Fargo 10-K Report)
2. Cross-reference EIN information in SEC filings
3. Extract key details such as dates signed and officer information from documents

## Project Structure

### Core Components

- **Workflow Engine**: Based on `pydantic-graph`, implementing a directed graph of processing nodes
- **Models**: Pydantic models defining structured data formats for extracted information
- **Services**: API integrations and processing capabilities
- **Config**: Configuration for the processing workflow

### Key Directories and Files

- `src/pdf_workflow/`: Main package containing all application code
  - `app.py`: Entry point for the application
  - `graph/`: Contains the workflow graph definition
  - `models/`: Data models for structured information extraction
  - `nodes/`: Processing nodes for the workflow graph
  - `config/`: Configuration settings
  - `services/`: External service integrations
  - `utils/`: Utility functions
  - `execution/`: Execution context management
  - `core/`: Core functionality
  - `entities/`: Entity definitions
  - `templates/`: Templates for various components

- `data/`: Directory for input and output data
- `docs/`: Project documentation
- `*.ps1`: PowerShell scripts for installation, running, and testing

## Technical Stack

- **Python**: 3.13
- **Key Dependencies**:
  - `google-genai`: Google's Generative AI API
  - `google-api-core`, `google-cloud-core`: Google Cloud services
  - `pydantic-graph`: Workflow graph implementation
- **Development Tools**:
  - `black`: Code formatting
  - `isort`: Import sorting
  - `pipenv`: Virtual environment and dependency management

## Processing Flow

The project implements a graph-based workflow with the following processing nodes:
1. `ConfigureAPI`: Sets up API configuration
2. `CreatePrompt`: Creates a prompt for LLM processing
3. `SanitizePrompt`: Sanitizes the input prompt
4. `ExecuteAPI`: Executes the API call to Gemini
5. `PrintResponse`: Displays the response
6. `ExportToCSV`: Exports extracted data to CSV format

## Data Models

The system uses a hierarchical data model with:
- `SecFiling`: Top-level document metadata
- `OcrTags`: Collection of extracted information
- Specific entity models:
  - `CompanyInfo`: Company details
  - `OfficerInfo`: Information about company officers
  - `EmployeeInfo`: Employee-related information
  - `CompanyAssets`: Company asset information

## Recent Improvements

- **Import Structure**: Converted relative imports to absolute imports throughout the codebase for better maintainability
- **Path Handling**: Implemented robust absolute path handling for credentials and data files
- **Installation**: Added proper Python package setup with setup.py and editable installation
- **Automation**: Created PowerShell scripts for development setup, running the application, and testing
- **Error Handling**: Enhanced error detection and reporting for file operations and API calls

## Current State and Next Steps

The project is a functional workflow for extracting specific information from PDF documents, with a focus on financial documents from the SEC. The codebase is well-structured with clear separation of concerns and improved reliability through better path handling and import structures.

Potential next steps might include:
- Expanding data extraction capabilities to additional document types
- Adding more validation and error handling
- Implementing batch processing capabilities
- Creating a web interface or API for interacting with the system
- Adding comprehensive testing 