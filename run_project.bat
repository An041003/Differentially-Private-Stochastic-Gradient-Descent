@echo off
setlocal EnableExtensions

REM One-click runner for the reproducible DP-SGD core experiment.
REM It creates .venv on first use, installs dependencies, then regenerates
REM processed data, CSV results, figures, report/report.{md,docx}, and slides.

cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher ^(py^) was not found.
    echo Install Python 3.12 or edit this file to point to your python.exe.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment with Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)

set "PYTHON=.venv\Scripts\python.exe"

echo [2/4] Installing required packages...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [3/4] Running the DP-SGD core experiment...
"%PYTHON%" scripts\run_final_experiments.py
if errorlevel 1 goto :failed

echo [4/4] Generating report and presentation from the new CSV results...
"%PYTHON%" scripts\generate_final_deliverables.py
if errorlevel 1 goto :failed

echo.
echo Completed successfully.
echo Results: results\
echo Figures: figures\
echo Report : report\report.md and report\report.docx
echo Slides : slides\dp_sgd_final_presentation.pptx
exit /b 0

:failed
echo.
echo [ERROR] The project stopped. Read the command output above for the cause.
exit /b 1
