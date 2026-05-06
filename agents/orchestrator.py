#!/usr/bin/env python3
"""
Autonomous Agent System: GPT-5 Mini (Planner) + Claude Opus 4.6 (Executor)

Fully automated loop:
1. GPT-5 Mini creates execution plan
2. Claude Opus 4.6 executes it (edits files, runs commands, etc.)
3. GPT-5 Mini verifies results
4. Loop until task is complete

Usage:
    $env:OPENAI_API_KEY = "sk-..."
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    py agents/orchestrator.py "sua tarefa aqui"

Requirements:
    pip install openai anthropic
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ─── API Clients ──────────────────────────────────────────────────────────────

try:
    import openai
    import anthropic
except ImportError:
    print("ERROR: Install dependencies first:")
    print("  pip install openai anthropic")
    sys.exit(1)


@dataclass
class AgentConfig:
    openai_key: str
    anthropic_key: str
    planner_model: str = "gpt-5-mini"  # GPT-5 Mini for planning
    executor_model: str = "claude-opus-4-20250514"  # Opus 4.6 for execution
    max_iterations: int = 10
    workspace: Path = Path(".")


# ─── Planner Agent (GPT-5 Mini) ───────────────────────────────────────────────

PLANNER_SYSTEM = """You are a senior software architect. Your ONLY job is to create detailed execution plans.

RULES:
1. Be EXTREMELY specific - include exact file paths, line numbers, code snippets
2. Each step must be independently executable
3. Include verification criteria for each step
4. Consider edge cases and error handling

OUTPUT FORMAT (strict JSON):
{
    "summary": "1-2 sentence overview",
    "steps": [
        {
            "id": 1,
            "action": "create_file" | "edit_file" | "run_command" | "read_file" | "delete_file",
            "target": "path/to/file or command",
            "content": "file content or edit description",
            "verification": "how to verify this step worked"
        }
    ],
    "success_criteria": "how to know the entire task is complete"
}

ONLY output valid JSON. No markdown, no explanations outside JSON."""


VERIFIER_SYSTEM = """You are a QA engineer verifying execution results.

Analyze the execution log and determine:
1. Was each planned step executed correctly?
2. Are there any errors or issues?
3. Is the task complete?

OUTPUT FORMAT (strict JSON):
{
    "status": "COMPLETE" | "NEEDS_CHANGES" | "FAILED",
    "completed_steps": [1, 2, 3],
    "failed_steps": [4],
    "issues": ["issue 1", "issue 2"],
    "changes_needed": [
        {
            "step_id": 4,
            "problem": "what went wrong",
            "fix": "specific fix instructions"
        }
    ],
    "message": "summary for human"
}

ONLY output valid JSON."""


def call_planner(config: AgentConfig, prompt: str) -> dict:
    """Call GPT-5 Mini for planning/verification."""
    client = openai.OpenAI(api_key=config.openai_key)
    
    response = client.chat.completions.create(
        model=config.planner_model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)


def call_verifier(config: AgentConfig, task: str, plan: dict, execution_log: str) -> dict:
    """Call GPT-5 Mini to verify execution."""
    client = openai.OpenAI(api_key=config.openai_key)
    
    prompt = f"""## Original Task
{task}

## Execution Plan
{json.dumps(plan, indent=2)}

## Execution Log
{execution_log}

Verify if the execution was successful."""
    
    response = client.chat.completions.create(
        model=config.planner_model,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2048,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)


# ─── Executor Agent (Claude Opus 4.6) ─────────────────────────────────────────

EXECUTOR_SYSTEM = """You are a code execution agent. Execute the given step EXACTLY as specified.

You have these capabilities:
- Read files
- Create/edit files
- Run shell commands
- Analyze code

For each step, execute it and report the result.

OUTPUT FORMAT (strict JSON):
{
    "success": true | false,
    "action_taken": "what you did",
    "output": "command output or file content preview",
    "error": "error message if failed, null if success"
}

ONLY output valid JSON."""


def execute_step_with_claude(config: AgentConfig, step: dict, workspace: Path) -> dict:
    """Have Claude execute a single step."""
    client = anthropic.Anthropic(api_key=config.anthropic_key)
    
    action = step.get("action", "")
    target = step.get("target", "")
    content = step.get("content", "")
    
    # Build execution prompt
    prompt = f"""Execute this step in workspace: {workspace}

Step ID: {step.get('id')}
Action: {action}
Target: {target}
Content/Details: {content}

Execute this step now and report the result."""
    
    # For some actions, we can execute directly without Claude
    result = {"success": False, "action_taken": "", "output": "", "error": None}
    
    try:
        if action == "create_file":
            file_path = workspace / target
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            result = {
                "success": True,
                "action_taken": f"Created file: {target}",
                "output": f"File created with {len(content)} chars",
                "error": None
            }
            
        elif action == "edit_file":
            # Use Claude to figure out the edit
            file_path = workspace / target
            if file_path.exists():
                current = file_path.read_text(encoding="utf-8")
            else:
                current = ""
            
            edit_prompt = f"""Current file content of {target}:
```
{current[:5000]}
```

Requested edit: {content}

Output the COMPLETE new file content. ONLY output the file content, nothing else."""
            
            response = client.messages.create(
                model=config.executor_model,
                max_tokens=8192,
                messages=[{"role": "user", "content": edit_prompt}]
            )
            new_content = response.content[0].text
            
            # Clean up markdown if present
            if new_content.startswith("```"):
                lines = new_content.split("\n")
                new_content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_content, encoding="utf-8")
            result = {
                "success": True,
                "action_taken": f"Edited file: {target}",
                "output": f"File updated, now {len(new_content)} chars",
                "error": None
            }
            
        elif action == "run_command":
            proc = subprocess.run(
                target,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=300
            )
            result = {
                "success": proc.returncode == 0,
                "action_taken": f"Ran command: {target}",
                "output": (proc.stdout + proc.stderr)[:2000],
                "error": None if proc.returncode == 0 else f"Exit code {proc.returncode}"
            }
            
        elif action == "read_file":
            file_path = workspace / target
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                result = {
                    "success": True,
                    "action_taken": f"Read file: {target}",
                    "output": content[:2000],
                    "error": None
                }
            else:
                result = {
                    "success": False,
                    "action_taken": f"Tried to read: {target}",
                    "output": "",
                    "error": "File not found"
                }
                
        elif action == "delete_file":
            file_path = workspace / target
            if file_path.exists():
                file_path.unlink()
                result = {
                    "success": True,
                    "action_taken": f"Deleted file: {target}",
                    "output": "File deleted",
                    "error": None
                }
            else:
                result = {
                    "success": True,
                    "action_taken": f"File already absent: {target}",
                    "output": "Nothing to delete",
                    "error": None
                }
                
        else:
            # Unknown action - ask Claude
            response = client.messages.create(
                model=config.executor_model,
                max_tokens=4096,
                system=EXECUTOR_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(response.content[0].text)
            
    except Exception as e:
        result = {
            "success": False,
            "action_taken": f"Attempted: {action} on {target}",
            "output": "",
            "error": str(e)
        }
    
    return result


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def run_autonomous_loop(config: AgentConfig, task: str):
    """Main autonomous loop: Plan → Execute → Verify → Repeat."""
    
    print("=" * 70)
    print("🤖 AUTONOMOUS AGENT SYSTEM")
    print("=" * 70)
    print(f"Planner: {config.planner_model}")
    print(f"Executor: {config.executor_model}")
    print(f"Workspace: {config.workspace.absolute()}")
    print(f"Task: {task}")
    print("=" * 70)
    print()
    
    iteration = 0
    execution_history = []
    
    while iteration < config.max_iterations:
        iteration += 1
        print(f"\n{'─'*60}")
        print(f"📋 ITERATION {iteration}/{config.max_iterations}")
        print(f"{'─'*60}")
        
        # ─── Step 1: Plan ─────────────────────────────────────────────
        print("\n🧠 [PLANNER] Creating execution plan...")
        
        context = ""
        if execution_history:
            context = f"\n\nPrevious execution history:\n{json.dumps(execution_history[-3:], indent=2)}"
        
        plan_prompt = f"""Task: {task}

Workspace: {config.workspace.absolute()}
{context}

Create a detailed execution plan."""
        
        try:
            plan = call_planner(config, plan_prompt)
        except Exception as e:
            print(f"❌ Planner error: {e}")
            continue
        
        print(f"   Summary: {plan.get('summary', 'N/A')}")
        print(f"   Steps: {len(plan.get('steps', []))}")
        
        # ─── Step 2: Execute ──────────────────────────────────────────
        print("\n⚡ [EXECUTOR] Running plan...")
        
        execution_log = []
        for step in plan.get("steps", []):
            step_id = step.get("id", "?")
            action = step.get("action", "unknown")
            target = step.get("target", "")
            
            print(f"   [{step_id}] {action}: {target[:50]}...", end=" ")
            
            result = execute_step_with_claude(config, step, config.workspace)
            execution_log.append({"step": step, "result": result})
            
            if result["success"]:
                print("✅")
            else:
                print(f"❌ {result.get('error', 'Unknown error')}")
        
        # ─── Step 3: Verify ───────────────────────────────────────────
        print("\n🔍 [VERIFIER] Checking results...")
        
        try:
            verification = call_verifier(
                config, task, plan,
                json.dumps(execution_log, indent=2)
            )
        except Exception as e:
            print(f"❌ Verifier error: {e}")
            continue
        
        status = verification.get("status", "UNKNOWN")
        message = verification.get("message", "No message")
        
        print(f"   Status: {status}")
        print(f"   Message: {message}")
        
        execution_history.append({
            "iteration": iteration,
            "plan_summary": plan.get("summary"),
            "steps_executed": len(execution_log),
            "verification_status": status,
            "issues": verification.get("issues", [])
        })
        
        # ─── Check if done ────────────────────────────────────────────
        if status == "COMPLETE":
            print("\n" + "=" * 70)
            print("✅ TASK COMPLETE!")
            print("=" * 70)
            print(f"Iterations: {iteration}")
            print(f"Final message: {message}")
            return True
        
        if status == "FAILED":
            print("\n" + "=" * 70)
            print("❌ TASK FAILED")
            print("=" * 70)
            print(f"Issues: {verification.get('issues', [])}")
            return False
        
        # NEEDS_CHANGES - continue loop
        changes = verification.get("changes_needed", [])
        print(f"   Changes needed: {len(changes)}")
        for change in changes:
            print(f"      - Step {change.get('step_id')}: {change.get('problem', 'Unknown')[:50]}")
        
        time.sleep(1)  # Rate limiting
    
    print("\n" + "=" * 70)
    print("⚠️ MAX ITERATIONS REACHED")
    print("=" * 70)
    return False


def main():
    parser = argparse.ArgumentParser(description="Autonomous Agent System")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--workspace", "-w", default=".", help="Workspace directory")
    parser.add_argument("--max-iter", "-n", type=int, default=10, help="Max iterations")
    parser.add_argument("--planner", default="gpt-5-mini", help="Planner model")
    parser.add_argument("--executor", default="claude-opus-4-20250514", help="Executor model")
    args = parser.parse_args()
    
    # Check API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)
    
    config = AgentConfig(
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        planner_model=args.planner,
        executor_model=args.executor,
        max_iterations=args.max_iter,
        workspace=Path(args.workspace).resolve()
    )
    
    success = run_autonomous_loop(config, args.task)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
