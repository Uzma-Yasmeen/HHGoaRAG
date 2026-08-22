@echo off
cd /d "%~dp0backend"
if not exist .venv\Scripts\python.exe (
  echo Backend environment is missing. Run setup-windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Starting backend. Keep this window open.
echo Health check: http://localhost:8000/health
python -m uvicorn app.main:app --reload --port 8000
pause
