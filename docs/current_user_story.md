# User Story: Async-First Codebase for Batch Processing

## Story

**As a** data analyst processing multiple SEC filings,  
**I want** the entire PDF Analyst workflow to operate in a fully asynchronous manner,  
**So that** I can efficiently process batches of files in parallel and maximize throughput.

## Business Value

By transforming the codebase to an async-first architecture:
- We can process batches of files in parallel, significantly reducing total processing time
- Improve resource utilization by not blocking during I/O operations
- Enhance scalability for higher volume document processing
- Support processing of larger datasets without performance degradation
- Enable more responsive user interfaces during long-running operations
- Provide better progress tracking and cancellation capabilities for batch operations

## Acceptance Criteria (Gherkin)

### Feature: Fully Asynchronous PDF Processing Workflow

#### Scenario 1: Process a batch of files concurrently
**Given** I have multiple PDF documents in the `data/input` directory  
**When** I run the PDF Analyst application in batch mode  
**Then** the files should be processed concurrently  
**And** the system should manage memory and resources efficiently  
**And** I should receive aggregated results for all processed files

#### Scenario 2: Monitor progress of batch processing
**Given** I am running the PDF Analyst application on a batch of files  
**When** the processing is underway  
**Then** I should receive real-time progress updates  
**And** see estimated completion time for the entire batch  
**And** be able to identify which files are currently being processed

#### Scenario 3: Cancel batch processing operations
**Given** I have started batch processing of PDF documents  
**When** I initiate a cancellation request  
**Then** the system should gracefully cancel all in-progress operations  
**And** preserve the results of already completed files  
**And** provide a summary of completed, canceled, and failed operations

#### Scenario 4: Handle errors without stopping the entire batch
**Given** I am processing a batch containing some problematic PDF files  
**When** the system encounters errors with specific files  
**Then** it should continue processing the remaining files  
**And** log detailed error information for the problematic files  
**And** provide a complete summary of successful and failed operations

## Technical Implementation Tasks

### 1. Refactor Remaining Nodes for Async Execution
- [ ] Update all workflow nodes to use consistent async patterns
  - [ ] Convert synchronous operations to async where appropriate
  - [ ] Implement proper exception handling in async context
  - [ ] Add cancellation support to all long-running operations
- [ ] Ensure all external API calls use async clients
  - [ ] Update Google API clients to async versions
  - [ ] Add timeouts and retry mechanisms for resiliency

### 2. Implement Batch Processing Controller
- [ ] Design BatchProcessor class to manage multiple file processing
  - [ ] Develop file discovery and validation mechanisms
  - [ ] Implement worker pool for parallel processing
  - [ ] Create batch-level metrics collection
- [ ] Build rate limiting and throttling mechanisms
  - [ ] Add configurable concurrency limits
  - [ ] Implement adaptive throttling based on system resources

### 3. Enhance GraphState for Batch Context
- [ ] Extend GraphState to support batch operations
  - [ ] Add batch identity and tracking information
  - [ ] Design aggregate result structures
  - [ ] Implement batch-level error handling
- [ ] Create progress monitoring capabilities
  - [ ] Build event system for progress updates
  - [ ] Add timing and estimation features

### 4. Update Application Interface
- [ ] Enhance CLI to support batch operations
  - [ ] Add batch-specific command line options
  - [ ] Implement interactive progress display
  - [ ] Create batch result summary reporting
- [ ] Update configuration handling
  - [ ] Add batch processing configuration options
  - [ ] Support per-file override configurations

### 5. Testing Infrastructure
- [ ] Update testing framework for async tests
  - [ ] Enhance PowerShell test scripts to handle async tests properly
  - [ ] Add batch-specific test scenarios
  - [ ] Implement mock batch data for testing
- [ ] Create performance testing suite
  - [ ] Build benchmarking tools for batch operations
  - [ ] Implement concurrency stress tests
  - [ ] Add resource utilization tracking to tests

### 6. Debug and Monitoring Tools
- [ ] Update local development scripts for async debugging
  - [ ] Create specialized debugging modes for async code
  - [ ] Add detailed logging for async operations
  - [ ] Implement visualization tools for async execution flow
- [ ] Enhance operational monitoring
  - [ ] Add batch operation metrics collection
  - [ ] Create dashboard for batch processing status
  - [ ] Implement alerting for batch failures

## Implementation Approach

The implementation will build on the foundation established with the `EncodeFileNode`, which already has async-compatible architecture:

1. **Gradual Migration**: Convert each component of the system to async patterns, starting with the most I/O-intensive operations.

2. **Unified Pattern**: Use consistent async patterns across the codebase, leveraging Python's `asyncio` library and modern async/await syntax.

3. **Resource Management**: Implement proper resource pools with configurable limits to prevent overloading the system during parallel processing.

4. **Error Isolation**: Ensure errors in one file's processing don't affect other files in the batch, with comprehensive error reporting.

5. **Test-Driven Approach**: Develop comprehensive async test cases before implementing each component to ensure correct behavior.

6. **Monitoring First**: Build detailed monitoring and progress reporting from the beginning to provide visibility into the async operations.

## Proposed Architecture

```python
# BatchProcessor.py
class BatchProcessor:
    """Manages batch processing of multiple files through the workflow."""
    
    def __init__(self, config: BatchConfig, max_concurrent: int = 5):
        self.config = config
        self.worker_pool = WorkerPool(max_concurrent)
        self.results = BatchResults()
    
    async def process_directory(self, input_dir: Path) -> BatchResults:
        """Process all valid files in a directory concurrently."""
        files = self._discover_files(input_dir)
        return await self.process_files(files)
    
    async def process_files(self, files: List[Path]) -> BatchResults:
        """Process a specific list of files concurrently."""
        tasks = [self._process_single_file(file) for file in files]
        return await self._gather_with_progress(tasks)
    
    async def _process_single_file(self, file_path: Path) -> FileResult:
        """Process a single file through the workflow."""
        # Implementation with proper error handling and metrics
```

## Definition of Done

- [ ] All components of the system operate asynchronously
- [ ] Batch processing capability is fully implemented and tested
- [ ] The system can efficiently handle multiple files concurrently
- [ ] Progress monitoring and cancellation support is available throughout
- [ ] Testing infrastructure supports async testing scenarios
- [ ] Debugging tools provide visibility into async execution
- [ ] Performance metrics demonstrate improved throughput for batch operations
- [ ] Documentation is updated to reflect the async architecture 