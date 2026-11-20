@echo off
rem Predatore V7 - misura le prestazioni su questo PC con 1, 4, 8 e 16 lavoratori.
cd /d "%~dp0"
python tools\benchmark_v7.py %*
pause
