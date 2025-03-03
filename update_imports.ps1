# Set the execution policy for this session to allow script execution
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Display start message
Write-Host "Starting import update for PDF Analyst..." -ForegroundColor Cyan

# Define the list of files to process
$filesToProcess = @(
    "src\pdf_workflow\app.py",
    "src\pdf_workflow\nodes\sanitize_prompt.py",
    "src\pdf_workflow\nodes\print_response.py",
    "src\pdf_workflow\nodes\export.py",
    "src\pdf_workflow\nodes\execute_api.py",
    "src\pdf_workflow\nodes\create_prompt.py",
    "src\pdf_workflow\nodes\configure_api.py",
    "src\pdf_workflow\models\sec_filing.py",
    "src\pdf_workflow\models\officer_info.py",
    "src\pdf_workflow\models\employee_info.py",
    "src\pdf_workflow\models\company_info.py",
    "src\pdf_workflow\models\company_assets.py",
    "src\pdf_workflow\graph\gemini_graph.py",
    "src\pdf_workflow\utils\model_armor\sanitize_prompt.py"
)

# Process each file
foreach ($file in $filesToProcess) {
    Write-Host "Processing $file..." -ForegroundColor Cyan
    
    # Check if the file exists
    if (Test-Path $file) {
        # Read the file content
        $content = Get-Content -Path $file -Raw
        
        # Replace import statements with double quotes for proper PowerShell regex
        $newContent = $content -replace 'from (config|graph|models|nodes|utils|execution|core|entities|templates)\.', 'from pdf_workflow.$1.'
        
        # Write the updated content back to the file
        Set-Content -Path $file -Value $newContent
        
        Write-Host "Updated $file" -ForegroundColor Green
    } else {
        Write-Host "File not found: $file" -ForegroundColor Yellow
    }
}

Write-Host "All import statements have been updated!" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 