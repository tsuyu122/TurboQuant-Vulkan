@echo off
REM Run autonomous agent system
REM Usage: run_agents.bat "sua tarefa aqui"

cd /d "C:\Users\hm\Desktop\TurboQuant vulkan"
call .venv\Scripts\activate.bat

REM Check if API keys are set
if "%OPENAI_API_KEY%"=="" (
    echo ERROR: Set OPENAI_API_KEY first
    echo Example: set OPENAI_API_KEY=sk-...
    exit /b 1
)
if "%ANTHROPIC_API_KEY%"=="" (
    echo ERROR: Set ANTHROPIC_API_KEY first  
    echo Example: set ANTHROPIC_API_KEY=sk-ant-...
    exit /b 1
)

python agents/orchestrator.py %*
