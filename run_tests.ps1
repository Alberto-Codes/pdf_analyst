# Testing and Coverage PowerShell Script for PDF Analyst
# This provides a complete solution for running tests and coverage

param (
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Set the execution policy for this session to allow script execution
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Create reports directory if it doesn't exist
if (-Not (Test-Path "reports")) {
    New-Item -ItemType Directory -Path "reports" | Out-Null
    Write-Host "Created reports directory" -ForegroundColor Green
}

# Check if pipenv is installed
try {
    $pipenvVersion = pipenv --version
    Write-Host "Found pipenv: $pipenvVersion" -ForegroundColor Green
} catch {
    Write-Host "Pipenv is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install pipenv with: pip install pipenv" -ForegroundColor Yellow
    exit 1
}

# Ensure development dependencies are installed
function Install-Dependencies {
    Write-Host "Checking development dependencies..." -ForegroundColor Cyan
    pipenv install --dev
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install development dependencies" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Load environment variables from .env file
function Load-EnvironmentVariables {
    if (Test-Path ".env") {
        Write-Host "Loading environment variables from .env file..." -ForegroundColor Cyan
        Get-Content ".env" | ForEach-Object {
            if (-not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#")) {
                $key, $value = $_ -split '=', 2
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
            }
        }
    }
    
    # Add test environment variables
    [Environment]::SetEnvironmentVariable("MODEL_ARMOR_PROJECT_ID", "test-project", "Process")
    [Environment]::SetEnvironmentVariable("MODEL_ARMOR_TEMPLATE_ID", "test-template", "Process")
    [Environment]::SetEnvironmentVariable("MODEL_ARMOR_MOCK", "true", "Process")
}

# Run EncodeFileNode unit tests
function Run-EncodeFileTests {
    Write-Host "Running EncodeFileNode unit tests..." -ForegroundColor Cyan
    $process = Start-Process -FilePath "pipenv" -ArgumentList "run", "python", "-m", "pytest", "tests/unit/pdf_workflow/nodes/test_encode_file.py", "-v" -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host "EncodeFileNode unit tests passed successfully!" -ForegroundColor Green
    } else {
        Write-Host "EncodeFileNode unit tests failed with exit code $($process.ExitCode)" -ForegroundColor Red
        exit $process.ExitCode
    }
}

# Run integration tests
function Run-IntegrationTests {
    Write-Host "Running workflow integration tests..." -ForegroundColor Cyan
    $process = Start-Process -FilePath "pipenv" -ArgumentList "run", "python", "-m", "pytest", "tests/unit/pdf_workflow/graph/test_gemini_graph.py", "-v" -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host "Workflow integration tests passed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Workflow integration tests failed with exit code $($process.ExitCode)" -ForegroundColor Red
        exit $process.ExitCode
    }
}

# Run all tests with coverage report
function Run-CoverageReport {
    Write-Host "Running all tests with coverage..." -ForegroundColor Cyan
    
    # Clean previous reports
    if (Test-Path "reports/coverage") {
        Write-Host "Cleaning previous coverage reports..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force "reports/coverage" | Out-Null
    }
    
    # Run tests with coverage
    $process = Start-Process -FilePath "pipenv" -ArgumentList "run", "python", "-m", "pytest", "tests/", "--cov=pdf_workflow", "--cov-report=term", "--cov-report=html:reports/coverage", "-v" -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host "All tests passed with coverage!" -ForegroundColor Green
        
        # Display coverage summary and open report
        if (Test-Path "reports/coverage/index.html") {
            Write-Host "Coverage report generated at: reports/coverage/index.html" -ForegroundColor Green
            Write-Host "Opening coverage report in browser..." -ForegroundColor Cyan
            Start-Process "reports/coverage/index.html"
        }
    } else {
        Write-Host "Tests failed with exit code $($process.ExitCode)" -ForegroundColor Red
        exit $process.ExitCode
    }
}

# Run all tests
function Run-AllTests {
    Install-Dependencies
    Load-EnvironmentVariables
    Run-EncodeFileTests
    Run-IntegrationTests
    Run-CoverageReport
    
    Write-Host "`nAll tests completed successfully!" -ForegroundColor Green
}

# Show help
function Show-Help {
    Write-Host ""
    Write-Host "PDF Analyst Testing Tools" -ForegroundColor Green
    Write-Host "------------------------" -ForegroundColor Green
    Write-Host "Usage: .\run_all_tests.ps1 [command]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Cyan
    Write-Host "  all              Run all tests and generate coverage report (default)"
    Write-Host "  encode-file      Run only the EncodeFileNode unit tests"
    Write-Host "  integration      Run only the workflow integration tests"
    Write-Host "  coverage         Run all tests with coverage and open report"
    Write-Host "  help             Display this help message"
    Write-Host ""
}

# Process command
switch ($Command.ToLower()) {
    "all" { Run-AllTests }
    "encode-file" { 
        Install-Dependencies
        Load-EnvironmentVariables
        Run-EncodeFileTests 
    }
    "integration" { 
        Install-Dependencies
        Load-EnvironmentVariables
        Run-IntegrationTests 
    }
    "coverage" { 
        Install-Dependencies
        Load-EnvironmentVariables
        Run-CoverageReport 
    }
    "help" { Show-Help }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor Yellow
        Write-Host "Running all tests by default..." -ForegroundColor Yellow
        Run-AllTests 
    }
} 