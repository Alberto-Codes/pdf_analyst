# Set the execution policy for this session to allow script execution
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Display start message
Write-Host "Setting up PDF Analyst test environment..." -ForegroundColor Cyan

# Check if pipenv is installed
try {
    $pipenvVersion = pipenv --version
    Write-Host "Found pipenv: $pipenvVersion" -ForegroundColor Green
} catch {
    Write-Host "Pipenv is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install pipenv with: pip install pipenv" -ForegroundColor Yellow
    exit 1
}

# Function to run a command in the pipenv environment
function Invoke-PipenvCommand {
    param (
        [string]$Command,
        [string]$Description
    )
    
    Write-Host "$Description..." -ForegroundColor Cyan
    pipenv run $Command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $Description completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "✗ $Description failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Ensure development dependencies are installed
try {
    Write-Host "Checking development dependencies..." -ForegroundColor Cyan
    pipenv install --dev
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install development dependencies" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "An error occurred during dependency installation:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Set environment variables from .env file if it exists
if (Test-Path ".env") {
    Write-Host "Loading environment variables from .env file..." -ForegroundColor Cyan
    Get-Content ".env" | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#")) {
            $key, $value = $_ -split '=', 2
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Run the tests
try {
    # Add test commands here
    Write-Host "No test commands configured yet." -ForegroundColor Yellow
    Write-Host "Please modify this script to add your test commands." -ForegroundColor Yellow
    
    # Example test commands (uncomment and modify as needed):
    # Invoke-PipenvCommand -Command "python -m pytest tests/" -Description "Running tests"
    # Invoke-PipenvCommand -Command "python -m pytest tests/ --cov=pdf_workflow" -Description "Running tests with coverage"
    
    Write-Host "All tasks completed!" -ForegroundColor Green
} catch {
    Write-Host "An error occurred:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Pause to view results
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 