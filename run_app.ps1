# Set the execution policy for this session to allow script execution
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Display start message
Write-Host "Starting PDF Analyst application..." -ForegroundColor Cyan

# Check if pipenv is installed
try {
    $pipenvVersion = pipenv --version
    Write-Host "Found pipenv: $pipenvVersion" -ForegroundColor Green
} catch {
    Write-Host "Pipenv is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install pipenv with: pip install pipenv" -ForegroundColor Yellow
    exit 1
}

# Get current directory for absolute path references
$currentDir = Get-Location

# Load environment variables from .env file
if (Test-Path ".env") {
    Write-Host "Loading environment variables from .env file..." -ForegroundColor Cyan
    
    Get-Content ".env" | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_) -and -not $_.StartsWith("#")) {
            $key, $value = $_ -split '=', 2
            
            # For GOOGLE_APPLICATION_CREDENTIALS, make sure it's an absolute path
            if ($key -eq "GOOGLE_APPLICATION_CREDENTIALS" -and -not [System.IO.Path]::IsPathRooted($value)) {
                $value = Join-Path -Path $currentDir -ChildPath $value
                Write-Host "Using absolute path for credentials: $value" -ForegroundColor Yellow
            }
            
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    
    # Verify the credentials file exists
    $credFile = [Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS")
    if (Test-Path $credFile) {
        Write-Host "Google credentials file found at: $credFile" -ForegroundColor Green
    } else {
        Write-Host "Warning: Google credentials file not found at: $credFile" -ForegroundColor Red
    }
}

# Run the app directly from the project root since we now use absolute imports
try {
    Write-Host "Running PDF Analyst application..." -ForegroundColor Cyan
    # Run the module directly instead of changing directory
    pipenv run python -m src.pdf_workflow.app

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Application executed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Application exited with code $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "An error occurred while running the application:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Simple end message
Write-Host ""
Write-Host "Script execution completed." -ForegroundColor Gray 