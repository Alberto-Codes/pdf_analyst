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

#### Scenario 1: Encode a PDF file in a dedicated node ✅
**Given** I have a PDF document located in the `data/input` directory  
**When** I run the PDF Analyst application  
**Then** the file should be processed by a dedicated `EncodeFileNode`  
**And** the encoded file content should be added to the workflow state  
**And** the processing should continue to the next node

#### Scenario 2: Handle large files efficiently ✅
**Given** I have a large PDF document to process  
**When** the `EncodeFileNode` processes the file  
**Then** it should manage memory efficiently during encoding  
**And** provide progress feedback for large files  
**And** continue the workflow without memory issues

#### Scenario 3: Error handling during file encoding ✅
**Given** I attempt to process an invalid or corrupted PDF file  
**When** the `EncodeFileNode` processes the file  
**Then** it should detect and handle the error gracefully  
**And** provide clear error messages about the failure  
**And** update the workflow state with error details

#### Scenario 4: Prepare for batch processing ✅
**Given** I have multiple PDF documents to process  
**When** I configure the application for batch processing  
**Then** the `EncodeFileNode` should be ready to handle sequential or parallel encoding  
**And** manage resources appropriately across multiple files

## Technical Implementation Tasks

### 1. Create New EncodeFileNode ✅
- [x] Design and implement `EncodeFileNode` class
  - [x] Add file validation and size checking
  - [x] Implement efficient file reading and encoding
  - [x] Handle various file encoding errors
  - [x] Support progress tracking for large files
- [x] Update `GraphState` to store encoded file content
  - [x] Add fields for tracking encoding status and metrics

### 2. Modify Existing Graph Structure ✅
- [x] Update `gemini_graph.py` to include the new encoding node
  - [x] Position `EncodeFileNode` between `ConfigureAPI` and `CreatePrompt`
  - [x] Ensure proper state handoff between nodes
- [x] Modify `CreatePrompt` to use pre-encoded content from state

### 3. Async Foundations ✅
- [x] Implement the node with async-compatible architecture
  - [x] Design helper methods with async signatures for future expansion
  - [x] Create extensible strategies for different file sizes
- [x] Add cancellation support for long-running encoding operations

### 4. Memory Optimization ✅
- [x] Define file size thresholds for different encoding strategies
- [x] Add hooks for future implementation of streaming file encoding for large files
- [x] Add memory usage monitoring through detailed metrics

### 5. Testing ✅
- [x] Create unit tests for the new encoding node
  - [x] Test with various file sizes and types
  - [x] Test error handling and edge cases
- [x] Add integration tests for the entire workflow with the new node
- [x] Set up comprehensive test automation
  - [x] Implement PowerShell test script with HTML coverage reporting
  - [x] Support for running specific test suites or all tests
- [x] Manual testing with the application ✅
  - Successfully tested with actual PDF file processing
  - Verified metrics collection and performance

### 6. Documentation ✅
- [x] Document the new node and its configuration options
- [x] Add clear explanation of encoding behavior and strategies
- [x] Provide comprehensive docstrings following Google Python style guide

## Implementation Notes

The implementation is now complete with the following key features:

1. **Dedicated EncodeFileNode**: The node validates file existence, reads the file content based on size strategy, and stores both encoded content and performance metrics in the GraphState.

2. **Error Handling**: Comprehensive error detection and handling for file operations with clear messages and metrics.

3. **Performance Tracking**: Timing and file size metrics are recorded to track encoding performance.

4. **Size-based Strategies**: The encoding operation adapts to file size (currently two strategies, with a 10MB threshold), providing a foundation for more sophisticated future optimizations.

5. **Logging**: Detailed logging throughout the encoding process for better monitoring and debugging.

6. **Async Foundation**: All encoding methods are designed for async compatibility, making future asynchronous processing straightforward to implement.

7. **Cancellation Support**: Implemented timeout-based cancellation for both small and large file encoding operations, ensuring long-running operations don't block the workflow indefinitely.

8. **Testing Coverage**: Comprehensive unit and integration tests for all aspects of the EncodeFileNode, including various file sizes, error handling, and metrics calculation.

## File Structure

The main implementation is organized as follows:

```python
# In encode_file.py
class EncodeFileNode(BaseNode[GraphState]):
    # File size threshold in MB for different encoding strategies
    SMALL_FILE_THRESHOLD_MB: float = 10.0
    
    # Timeout for encoding operations in seconds
    ENCODING_TIMEOUT_SECONDS: float = 60.0
    
    async def run(self, ctx: GraphRunContext[GraphState]) -> CreatePrompt:
        # Implementation of the encoding workflow
        # ...
    
    async def _encode_small_file(self, filepath: Path) -> bytes:
        # Method for handling small files with cancellation support
        # ...
    
    async def _encode_large_file(self, filepath: Path) -> bytes:
        # Method for handling large files with cancellation support
        # ...
    
    def _finish_encoding_metrics(self, metrics: Dict[str, Any]) -> None:
        # Updates the metrics after encoding completes
        # ...
```

## Definition of Done

- [x] `EncodeFileNode` is implemented and integrated into the workflow
- [x] File encoding is performed efficiently for various file sizes
- [x] The node provides proper error handling and reporting
- [x] The architecture supports future async operations
- [x] Tests verify correct behavior in all scenarios
- [x] Documentation is updated to reflect the new workflow architecture
- [x] The solution is more maintainable than the previous approach 