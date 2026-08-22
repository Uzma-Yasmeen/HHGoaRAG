@echo off
cd /d "%~dp0"
echo Starting website. Keep this window open.
call npm run dev
pause
