#!/usr/bin/env python3
"""
MCP Server: Claude Planner Agent

Exposes tools for a planning/verification loop with Claude API:
- plan_task: Send a task to Claude, get a structured plan
- verify_result: Have Claude verify if execution result is correct
- request_changes: Get specific change requests from Claude

Usage:
    Configure in VS Code settings.json:
    "mcp": {
        "servers": {
            "planner": {
                "command": "py",
                "args": ["-u", "mcp_planner/server.py"],
                "cwd": "C:\\Users\\hm\\Desktop\\TurboQuant vulkan"
            }
        }
    }

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import sys
import os
from typing import Any

# Anthropic API
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ─── MCP Protocol ─────────────────────────────────────────────────────────────

def read_message() -> dict | None:
    """Read a JSON-RPC message from stdin."""
    try:
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def write_message(msg: dict):
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def make_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def make_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ─── Claude API ───────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a senior software architect and planner. Your role is to:
1. Analyze tasks and break them into clear, actionable steps
2. Provide specific file paths and code changes needed
3. Anticipate edge cases and potential issues

When creating a plan:
- Be SPECIFIC about files to create/modify
- Include exact code snippets when possible
- Number each step clearly
- Mark dependencies between steps

Format your response as:
## Plan Summary
[1-2 sentence overview]

## Steps
1. **[Action]**: [Details]
   - File: `path/to/file`
   - Change: [specific change]
   
2. **[Action]**: [Details]
   ...

## Verification Criteria
- [How to verify step 1 worked]
- [How to verify step 2 worked]
"""

VERIFIER_SYSTEM = """You are a code reviewer verifying execution results. Your role is to:
1. Check if the execution matched the plan
2. Identify any errors or issues
3. Determine if the task is complete or needs changes

Respond with:
## Status
[COMPLETE | NEEDS_CHANGES | FAILED]

## Analysis
[What was done correctly]

## Issues (if any)
[Specific problems found]

## Required Changes (if NEEDS_CHANGES)
1. [Specific change needed]
2. [Specific change needed]
"""


def call_claude(system: str, user: str) -> str:
    """Call Claude API and return response text."""
    if not HAS_ANTHROPIC:
        return "ERROR: anthropic package not installed. Run: pip install anthropic"
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY not set in environment"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text
    except Exception as e:
        return f"ERROR: Claude API call failed: {e}"


# ─── Tool Implementations ─────────────────────────────────────────────────────

def tool_plan_task(task_description: str, context: str = "") -> str:
    """Create a detailed execution plan for a task."""
    prompt = f"""Task to plan:
{task_description}

{"Context/Background:" + chr(10) + context if context else ""}

Create a detailed, actionable plan that another AI agent can execute step by step.
Include specific file paths, code snippets, and verification criteria."""
    
    return call_claude(PLANNER_SYSTEM, prompt)


def tool_verify_result(original_task: str, plan: str, execution_result: str) -> str:
    """Verify if execution result matches the plan."""
    prompt = f"""## Original Task
{original_task}

## Plan That Was Executed
{plan}

## Execution Result
{execution_result}

Analyze whether the execution was successful and complete. If changes are needed, be specific about what to fix."""
    
    return call_claude(VERIFIER_SYSTEM, prompt)


def tool_request_changes(issue_description: str, current_state: str) -> str:
    """Get specific change requests to fix an issue."""
    prompt = f"""## Issue
{issue_description}

## Current State
{current_state}

Provide specific, actionable changes to fix this issue. Include exact code/commands."""
    
    system = """You are a debugging expert. Provide specific, minimal changes to fix the described issue.
Format as numbered steps with exact code snippets or commands."""
    
    return call_claude(system, prompt)


# ─── MCP Handlers ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "plan_task",
        "description": "Send a task to the Claude planner agent. Returns a detailed execution plan with specific steps, file paths, and code changes. Use this FIRST when starting a complex task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Detailed description of what needs to be done"
                },
                "context": {
                    "type": "string",
                    "description": "Optional: relevant context, file contents, error messages"
                }
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "verify_result",
        "description": "Have Claude verify if your execution result is correct. Use this AFTER completing the plan steps to check if everything worked.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "original_task": {
                    "type": "string",
                    "description": "The original task that was requested"
                },
                "plan": {
                    "type": "string",
                    "description": "The plan that was executed"
                },
                "execution_result": {
                    "type": "string",
                    "description": "Summary of what was done and any outputs/errors"
                }
            },
            "required": ["original_task", "plan", "execution_result"]
        }
    },
    {
        "name": "request_changes",
        "description": "Get specific change requests from Claude to fix an issue. Use this when verification found problems.",
        "inputSchema": {
            "type": "object", 
            "properties": {
                "issue_description": {
                    "type": "string",
                    "description": "What went wrong or needs to be fixed"
                },
                "current_state": {
                    "type": "string",
                    "description": "Current state of the code/system"
                }
            },
            "required": ["issue_description", "current_state"]
        }
    }
]


def handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "claude-planner", "version": "1.0.0"}
    }


def handle_tools_list() -> dict:
    return {"tools": TOOLS}


def handle_tools_call(params: dict) -> dict:
    name = params.get("name")
    args = params.get("arguments", {})
    
    if name == "plan_task":
        result = tool_plan_task(
            args.get("task_description", ""),
            args.get("context", "")
        )
    elif name == "verify_result":
        result = tool_verify_result(
            args.get("original_task", ""),
            args.get("plan", ""),
            args.get("execution_result", "")
        )
    elif name == "request_changes":
        result = tool_request_changes(
            args.get("issue_description", ""),
            args.get("current_state", "")
        )
    else:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
    
    return {"content": [{"type": "text", "text": result}]}


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    # Unbuffered stderr for logging
    sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)
    
    while True:
        msg = read_message()
        if msg is None:
            break
        
        method = msg.get("method")
        params = msg.get("params", {})
        msg_id = msg.get("id")
        
        try:
            if method == "initialize":
                result = handle_initialize(params)
            elif method == "tools/list":
                result = handle_tools_list()
            elif method == "tools/call":
                result = handle_tools_call(params)
            elif method == "notifications/initialized":
                continue  # No response needed
            else:
                if msg_id is not None:
                    write_message(make_error(msg_id, -32601, f"Unknown method: {method}"))
                continue
            
            if msg_id is not None:
                write_message(make_response(msg_id, result))
                
        except Exception as e:
            if msg_id is not None:
                write_message(make_error(msg_id, -32603, str(e)))


if __name__ == "__main__":
    main()
