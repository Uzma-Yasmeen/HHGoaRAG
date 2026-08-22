@echo off
cd /d "%~dp0"
echo Checking GoaVaani backend at http://localhost:8000/health
powershell -NoProfile -Command "try { $h=Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 8; $h | ConvertTo-Json; if(-not $h.stt_ready){Write-Host ''; Write-Host 'VOICE NOT READY: add ELEVENLABS_API_KEY to backend\.env and restart run-local.bat' -ForegroundColor Yellow} } catch { Write-Host 'BACKEND NOT REACHABLE: run setup-windows.bat once, then run-local.bat.' -ForegroundColor Red; Write-Host $_.Exception.Message }"
echo.
pause
