@echo off
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
python scripts\benchmark.py --runs 100 --warmup 10 --mode fast --output benchmark-results-fast.json
if errorlevel 1 exit /b %errorlevel%
copy /Y benchmark-results-fast.json ..\public\benchmark-results-fast.json >nul
echo Published results to public\benchmark-results-fast.json
pause
