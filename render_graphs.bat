@echo off
setlocal enabledelayedexpansion

REM ==============================
REM Firmware Graphviz Renderer
REM ==============================

REM Check dot existence
where dot >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Graphviz 'dot' not found in PATH.
    echo Please install Graphviz and add it to PATH.
    pause
    exit /b
)

echo.
echo ============================================
echo  Firmware Flow Graph Renderer
echo ============================================
echo.

REM ===== Choose output format =====
set FORMAT=png
REM You can change to svg if needed:
REM set FORMAT=svg

echo Output format: %FORMAT%
echo.

REM ===== Directories =====
set FLOW_SRC=output\flow_v2\dot_flow
set ARCH_SRC=output\flow_v2\dot_arch

set FLOW_OUT=output\flow_v2\images_flow
set ARCH_OUT=output\flow_v2\images_arch

if not exist "%FLOW_OUT%" mkdir "%FLOW_OUT%"
if not exist "%ARCH_OUT%" mkdir "%ARCH_OUT%"

echo Rendering FLOW graphs...
for %%f in (%FLOW_SRC%\*.dot) do (
    echo   Converting %%~nxf
    dot -T%FORMAT% "%%f" -o "%FLOW_OUT%\%%~nf.%FORMAT%"
)

echo.
echo Rendering ARCH graphs...
for %%f in (%ARCH_SRC%\*.dot) do (
    echo   Converting %%~nxf
    dot -T%FORMAT% "%%f" -o "%ARCH_OUT%\%%~nf.%FORMAT%"
)

echo.
echo ============================================
echo  DONE
echo ============================================
echo.
echo Flow images: %FLOW_OUT%
echo Arch images: %ARCH_OUT%
echo.
pause