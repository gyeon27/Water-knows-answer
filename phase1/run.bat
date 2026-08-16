@echo off
REM 더블클릭하면 실행됩니다. C:\venvs\fluid_routing 에 만들어둔 파이썬(가상환경)으로
REM 이 폴더(한글 경로)의 main.py를 실행합니다.
REM 주의: venv를 한글 경로(예: 이 프로젝트 폴더) 안에 두면 Taichi GGUI가 셰이더 리소스를
REM 못 읽어서 scene.particles() 호출 시 RuntimeError가 발생한다. 그래서 venv는 반드시
REM ASCII 경로(C:\venvs\...)에 둔다.
cd /d "%~dp0"
C:\venvs\fluid_routing\Scripts\python.exe main.py
pause
