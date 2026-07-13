@echo off
REM ------------------------------------------------------------
REM  compile.bat - build the Beamer deck with Tectonic.
REM  Double-click to run, or from a terminal:  compile.bat [file.tex]
REM  Defaults to presentation.tex in this folder.
REM ------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "DOC=%~1"
if "%DOC%"=="" set "DOC=presentation.tex"

REM Prefer tectonic on PATH; fall back to the conda 'tex' env.
set "TECTONIC=tectonic"
where tectonic >nul 2>nul || set "TECTONIC=C:\Users\kaima\miniconda3\envs\tex\Library\bin\tectonic.exe"

echo Compiling %DOC% with Tectonic...
echo.
"%TECTONIC%" --synctex --keep-logs "%DOC%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" echo Build succeeded: %DOC:.tex=.pdf%
if not "%RC%"=="0" echo Build FAILED (exit %RC%). See the .log for details.

REM Pause only when launched by double-click, not from a terminal.
echo %CMDCMDLINE% | findstr /i /c:"%~nx0" >nul && pause
endlocal & exit /b %RC%
