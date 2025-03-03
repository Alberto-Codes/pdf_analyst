# Set the execution policy for this session to allow script execution
# This won't permanently change your system settings
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Display start message
Write-Host "Starting development installation for PDF Analyst..." -ForegroundColor Cyan

# Check if pipenv is installed
try {
    $pipenvVersion = pipenv --version
    Write-Host "Found pipenv: $pipenvVersion" -ForegroundColor Green
} catch {
    Write-Host "Pipenv is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install pipenv with: pip install pipenv" -ForegroundColor Yellow
    exit 1
}

# Install the package in development mode
try {
    Write-Host "Installing PDF Analyst in development mode..." -ForegroundColor Cyan
    pipenv install -e .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installation completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run your code using: pipenv run python -m pdf_workflow.app" -ForegroundColor Cyan
        Write-Host "Or start a shell with: pipenv shell" -ForegroundColor Cyan
    } else {
        Write-Host "Installation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "An error occurred during installation:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Pause to view results
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 