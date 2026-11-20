@echo off
cd /d "%~dp0"
python launcher.py --gui
if errorlevel 1 pause
