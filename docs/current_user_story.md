# User Story: Direct File Integration with Gemini API

## Story

**As a** data analyst working with SEC filings,  
**I want to** upload PDF files directly to the Gemini API instead of encoding them,  
**So that** I can process larger documents more efficiently and reduce the complexity of the application.

## Business Value

By implementing direct file upload and management through the Gemini API's file features, we can:
- Handle larger documents that might exceed encoding limits
- Improve processing speed by avoiding base64 encoding/decoding overhead
- Simplify the codebase by delegating file management to the API
- Reduce memory usage during document processing

## Acceptance Criteria (Gherkin)

### Feature: PDF File Upload and Processing via Gemini API

#### Scenario 1: Upload a PDF file to Gemini API
**Given** I have a PDF document located in the `data/input` directory  
**When** I run the PDF Analyst application  
**Then** the file should be uploaded to Gemini API  
**And** the application should receive a file ID  
**And** the file ID should be stored for future reference

#### Scenario 2: Process an uploaded file
**Given** I have uploaded a PDF document to Gemini API  
**When** I execute the processing workflow  
**Then** the application should reference the file by ID instead of encoding it  
**And** the Gemini API should extract information according to the prompt  
**And** the extracted information should be returned in the expected format

#### Scenario 3: Delete a processed file
**Given** I have processed a PDF document using the Gemini API  
**When** the processing workflow completes  
**Then** the application should delete the file from Gemini API  
**And** confirm the deletion was successful

#### Scenario 4: Handle upload failures
**Given** I attempt to upload an invalid or corrupted PDF file  
**When** the upload operation is executed  
**Then** the application should handle the error gracefully  
**And** provide clear error messages about the failure

## Technical Implementation Tasks

### 1. Add New Graph Nodes
- [ ] Create `UploadFileNode` to handle file uploads to Gemini API
  - Implement file validation before upload
  - Handle file path resolution (relative vs. absolute)
  - Return file ID for subsequent operations
- [ ] Create `DeleteFileNode` to clean up files after processing
  - Add configurable option to retain files for debugging
  - Implement proper error handling for deletion failures

### 2. Modify Existing Graph Structure
- [ ] Update `gemini_graph.py` to include new file handling nodes
  - Add `UploadFileNode` before `CreatePrompt`
  - Add `DeleteFileNode` after `ExportToCSV`
- [ ] Modify state object to track file IDs and statuses

### 3. API Integration
- [ ] Implement Gemini file API client integration
  - Use `client.files.upload()` method for uploading files
  - Use `client.files.delete()` method for cleanup
- [ ] Add proper authentication and error handling
- [ ] Implement retry logic for transient API failures

### 4. Update Prompt Structure
- [ ] Modify prompt templates to reference uploaded files
  - Replace base64 encoding with file references
  - Update schema to support file ID references

### 5. Testing
- [ ] Create unit tests for new file handling nodes
- [ ] Add integration tests for the entire workflow with file uploading
- [ ] Test edge cases (large files, invalid files, API failures)

### 6. Documentation
- [ ] Update project documentation to reflect new file handling approach
- [ ] Add examples of using the new file upload capabilities
- [ ] Document failure scenarios and how they're handled

## Implementation Notes

### Gemini Files API
Based on the Gemini API documentation, we'll use the following methods:

```python
# Upload a file
file = client.files.upload(
    path="/path/to/file.pdf",  # Local file path
    display_name="SEC Filing",  # Optional display name
    mime_type="application/pdf"  # Specify MIME type
)

# Use the file in a prompt
response = client.generate_content(
    contents=[
        {
            "file_data": {
                "file_uri": file.uri,
                "mime_type": "application/pdf"
            }
        },
        "Extract company information from this document"
    ]
)

# Delete the file when done
client.files.delete(name=file.name)
```

### Pydantic Graph Integration
We'll integrate with the existing pydantic-graph workflow:

```python
# New file upload node
class UploadFileNode(BaseNode[GraphState]):
    def run(self, ctx: GraphRunContext[GraphState]):
        file_path = ctx.state.document_path
        file = ctx.state.client.files.upload(
            path=file_path,
            mime_type=ctx.state.document_mime_type
        )
        ctx.state.file_id = file.name
        ctx.state.file_uri = file.uri
        return CreatePrompt()

# Modified graph definition
gemini_graph = Graph(
    [
        ConfigureAPI,
        UploadFileNode,  # New node
        CreatePrompt,
        SanitizePrompt,
        ExecuteAPI,
        PrintResponse,
        ExportToCSV,
        DeleteFileNode   # New node
    ]
)
```

## Definition of Done
- All tasks are completed and code is reviewed
- All tests pass, including edge cases
- Documentation is updated
- The application can process documents of all supported sizes
- File cleanup is reliable, with no orphaned files
- Performance metrics show improvement over the encoding-based approach 