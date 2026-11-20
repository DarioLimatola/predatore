@echo off
rem Predatore V7 - compila e installa il nucleo nativo Rust (predatore_native).
rem Serve una sola volta, o dopo un aggiornamento della cartella native\.
rem Requisiti: Python 3.10+, Rust (https://rustup.rs) con gli strumenti C++ di Visual Studio
rem (rustup-init li propone durante l'installazione).
cd /d "%~dp0"
where cargo >nul 2>nul
if errorlevel 1 (
  echo Rust non trovato. Installa Rust da https://rustup.rs e riapri questa finestra.
  pause
  exit /b 1
)
python -m pip install --upgrade "maturin>=1.7,<2"
if errorlevel 1 goto fail
python -m maturin build --release --manifest-path native\Cargo.toml --out native\dist
if errorlevel 1 goto fail
for %%f in (native\dist\predatore_native-*.whl) do python -m pip install --force-reinstall --no-deps "%%f"
if errorlevel 1 goto fail
python launcher.py --engine-status
echo.
echo Nucleo nativo installato. Avvia il laboratorio con AVVIA_PREDATORE_V7.bat
pause
exit /b 0
:fail
echo Compilazione non riuscita. Il laboratorio funziona comunque con il motore Python.
pause
exit /b 1
