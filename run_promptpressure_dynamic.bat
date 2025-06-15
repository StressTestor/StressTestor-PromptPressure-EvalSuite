@echo off
setlocal enabledelayedexpansion

:: Save the current directory
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

echo [INFO] Running PromptPressure Eval Suite from: %CD%

:: Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please ensure Python is installed and available in your system PATH.
    pause
    exit /b 1
)

:: Try to activate virtual environment if it exists
if exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Failed to activate virtual environment. Continuing with system Python.
    )
)

echo [INFO] Starting PromptPressure...
python run_eval.py --config "%SCRIPT_DIR%config.yaml"
set "EXIT_CODE=!ERRORLEVEL!"

if !EXIT_CODE! EQU 0 (
    echo [INFO] PromptPressure completed successfully.
) else (
    echo [ERROR] PromptPressure failed with exit code !EXIT_CODE!.
)

pause
popd
exit /b %EXIT_CODE%
