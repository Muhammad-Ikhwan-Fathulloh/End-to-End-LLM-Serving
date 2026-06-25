@echo off
echo Starting Smart Manufacturing Blueprint Analyzer Backend...
cd "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in backend/venv.
    echo Please follow the setup instructions in README.md first.
    pause
    exit /b 1
)
.\venv\Scripts\uvicorn.exe main:app --reload --host 0.0.0.0 --port 8000
pause
