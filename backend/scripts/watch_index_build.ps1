param([Parameter(Mandatory = $true)][int]$BuildPid)

$backendPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$statusLog = Join-Path $backendPath 'index-build.status.log'
$indexPath = Join-Path $backendPath 'data\indexes'
$required = @('en.faiss', 'en.jsonl', 'hi.faiss', 'hi.jsonl', 'te.faiss', 'te.jsonl')

"Waiting for index build process $BuildPid" | Set-Content -LiteralPath $statusLog
Wait-Process -Id $BuildPid -ErrorAction SilentlyContinue

$missing = $required | Where-Object {
    $path = Join-Path $indexPath $_
    -not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -eq 0
}

if ($missing) {
    "Index build did not complete. Missing: $($missing -join ', ')" | Add-Content -LiteralPath $statusLog
    exit 1
}

"All index artifacts verified. Restarting backend." | Add-Content -LiteralPath $statusLog
$listeners = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
}
if ($listeners) { Start-Sleep -Milliseconds 750 }
$python = Join-Path $backendPath '.venv\Scripts\python.exe'
$stdout = Join-Path $backendPath 'backend.stdout.log'
$stderr = Join-Path $backendPath 'backend.stderr.log'
$backend = Start-Process -FilePath $python -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--port', '8000') -WorkingDirectory $backendPath -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
"Backend started as process $($backend.Id)." | Add-Content -LiteralPath $statusLog
