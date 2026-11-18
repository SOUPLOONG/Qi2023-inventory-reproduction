@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_full.py --profile quick --stage all
) else (
  python run_full.py --profile quick --stage all
)
if errorlevel 1 echo 运行失败，请查看上方错误。依赖安装步骤见README.md。
pause
