@echo off
REM GGUI 창 없이 300프레임만 돌려서 fluid_routing_log.csv / fluid_routing_summary.json 을 만듭니다.
cd /d "%~dp0"
C:\venvs\fluid_routing\Scripts\python.exe main.py --headless --max-frames 300
pause
