# User Story: Dedicated File Encoding Node for Async Processing

## Story

**As a** data analyst working with SEC filings,  
**I want to** have PDF file encoding in a dedicated workflow node,  
**So that** I can optimize the processing pipeline for async operations and batch processing in the future.

## Business Value

By implementing a dedicated file encoding node in the workflow:
- We can isolate resource-intensive encoding operations from other processing steps
- Prepare the architecture for future asynchronous processing capabilities
- Enable better memory management during batch processing of multiple documents
- Provide a clear separation of concerns in the workflow, making it more maintainable
- Support better error handling and retries specific to file handling

## Acceptance Criteria (Gherkin)

### Feature: Dedicated File Encoding Node in PDF Processing Workflow

#### Scenario 1: Encode a PDF file in a dedicated node
**Given** I have a PDF document located in the `data/input` directory  
**When** I run the PDF Analyst application  
**Then** the file should be processed by a dedicated `EncodeFileNode`  
**And** the encoded file content should be added to the workflow state  
**And** the processing should continue to the next node

#### Scenario 2: Handle large files efficiently
**Given** I have a large PDF document to process  
**When** the `EncodeFileNode` processes the file  
**Then** it should manage memory efficiently during encoding  
**And** provide progress feedback for large files  
**And** continue the workflow without memory issues

#### Scenario 3: Error handling during file encoding
**Given** I attempt to process an invalid or corrupted PDF file  
**When** the `EncodeFileNode` processes the file  
**Then** it should detect and handle the error gracefully  
**And** provide clear error messages about the failure  
**And** update the workflow state with error details

#### Scenario 4: Prepare for batch processing
**Given** I have multiple PDF documents to process  
**When** I configure the application for batch processing  
**Then** the `EncodeFileNode` should be ready to handle sequential or parallel encoding  
**And** manage resources appropriately across multiple files

## Technical Implementation Tasks

### 1. Create New EncodeFileNode
- [ ] Design and implement `EncodeFileNode` class
  - Add file validation and size checking
  - Implement efficient file reading and encoding
  - Handle various file encoding errors
  - Support progress tracking for large files
- [ ] Update `GraphState` to store encoded file content
  - Add fields for tracking encoding status and metrics

### 2. Modify Existing Graph Structure
- [ ] Update `gemini_graph.py` to include the new encoding node
  - Position `EncodeFileNode` between `ConfigureAPI` and `CreatePrompt`
  - Ensure proper state handoff between nodes
- [ ] Modify `CreatePrompt` to use pre-encoded content from state

### 3. Async Foundations
- [ ] Implement the node with async-compatible architecture
  - Ensure the encoding operation can be processed asynchronously in the future
  - Add cancellation support for long-running encoding operations
- [ ] Add resource management hooks for future parallel processing

### 4. Memory Optimization
- [ ] Implement streaming file encoding for large files
  - Use buffer-based approaches for large files
  - Add file size thresholds for different encoding strategies
- [ ] Add memory usage monitoring during encoding

### 5. Testing
- [ ] Create unit tests for the new encoding node
  - Test with various file sizes and types
  - Test error handling and edge cases
- [ ] Add integration tests for the entire workflow with the new node
- [ ] Benchmark performance with different file sizes

### 6. Documentation
- [ ] Document the new node and its configuration options
- [ ] Add examples of customizing encoding behavior
- [ ] Update workflow diagrams to include the new node

## Implementation Notes

### File Encoding Approach
We'll implement efficient file encoding with the following considerations:

```python
# New encode file node
class EncodeFileNode(BaseNode[GraphState]):
    async def run(self, ctx: GraphRunContext[GraphState]) -> CreatePrompt:
        # Validate file existence
        filepath = Path(ctx.state.document_path)
        if not filepath.exists():
            ctx.state.error = {
                "type": "file_error",
                "message": "File not found",
                "details": f"The file at {filepath} does not exist"
            }
            return CreatePrompt()
            
        # Check file size and choose encoding strategy
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Encoding file: {filepath} (Size: {file_size_mb:.2f} MB)")
        
        try:
            # For smaller files, read all at once
            if file_size_mb < 10:
                file_bytes = filepath.read_bytes()
            else:
                # For larger files, consider chunked reading in future
                # This is a placeholder for future optimization
                file_bytes = filepath.read_bytes()
                
            # Store encoded content in state
            ctx.state.encoded_file = {
                "bytes": file_bytes,
                "mime_type": ctx.state.document_mime_type,
                "size": len(file_bytes)
            }
            
            print(f"Successfully encoded {len(file_bytes)} bytes")
            return CreatePrompt()
            
        except Exception as e:
            ctx.state.error = {
                "type": "encoding_error",
                "message": "Failed to encode file",
                "details": str(e)
            }
            return CreatePrompt()
```

### GraphState Modifications
We'll extend the `GraphState` model to include encoding information:

```python
class GraphState(BaseModel):
    # Existing fields...
    
    # New fields for encoded file
    encoded_file: Dict[str, Any] = None
    encoding_metrics: Dict[str, Any] = None
```

### Modified Graph Definition
The workflow will be updated to include the new encoding node:

```python
gemini_graph = Graph(
    nodes=[
        ConfigureAPI,        # Returns EncodeFileNode
        EncodeFileNode,      # Returns CreatePrompt
        CreatePrompt,        # Returns SanitizePrompt
        SanitizePrompt,      # Returns ExecuteAPI
        ExecuteAPI,          # Returns PrintResponse
        PrintResponse,       # Returns ExportToCSV
        ExportToCSV          # Returns End
    ]
)
```

## Definition of Done
- `EncodeFileNode` is implemented and integrated into the workflow
- File encoding is performed efficiently for various file sizes
- The node provides proper error handling and reporting
- The architecture supports future async operations
- Tests verify correct behavior in all scenarios
- Documentation is updated to reflect the new workflow architecture
- The solution is more maintainable than the previous approach 