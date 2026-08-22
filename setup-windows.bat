@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Checking Node.js...
node -e "const m=+process.versions.node.split('.')[0]; if(m<22){console.error('Node.js 22 or newer is required. Install it from https://nodejs.org');process.exit(1)}"
if errorlevel 1 (
  pause
  exit /b 1
)
echo [2/3] Installing website packages...
call npm install
if errorlevel 1 (
  pause
  exit /b 1
)
echo [3/3] Creating Python environment...
cd backend
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Add your keys to backend\.env, then run run-local.bat
pause
