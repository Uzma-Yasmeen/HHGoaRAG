@echo off
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
python scripts\build_index.py --languages en hi te --limit-per-language 20000
echo Restart the backend after the index finishes.
pause
