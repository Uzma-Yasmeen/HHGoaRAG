@echo off
cd /d "%~dp0"
if not exist backend\.venv\Scripts\python.exe (
  echo Run setup-windows.bat first.
  pause
  exit /b 1
)
start "GoaVaani Backend" cmd /k "cd /d ""%~dp0backend"" && call .venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --port 8000"
start "GoaVaani Website" cmd /k "cd /d ""%~dp0"" && npm run dev"
timeout /t 4 /nobreak >nul
start http://localhost:5173
