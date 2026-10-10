
import os
import re
import json
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Code Interpreter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# The request must contain a Python code string.
class CodeRequest(BaseModel):
    code: str


# The AI response must contain a list of line numbers.
class ErrorAnalysis(BaseModel):
    error_lines: list[int] = Field(default_factory=list)


def execute_python_code(code: str) -> dict:
    """Execute Python code and capture its output or traceback."""
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compile(code, "<string>", "exec"), {})

        output = stdout_buffer.getvalue() + stderr_buffer.getvalue()

        return {
            "success": True,
            "output": output,
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }


def analyze_error_with_ai(code: str, error_traceback: str) -> list[int]:
    """Ask AIPipe to identify the relevant source-code line numbers."""
    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set.")

    prompt = f"""
Analyze this Python program and its traceback.

Return JSON with exactly this field:
{{"error_lines": [3]}}

Rules:
- Identify the line numbers in the submitted Python code that caused
  the error.
- Count lines starting at 1.
- Return only line numbers that exist in the submitted code.
- If the exact line cannot be determined, return an empty list.
- Do not return traceback line numbers that refer to Python libraries.

Python code:
<code>
{code}
</code>

Traceback:
<traceback>
{error_traceback}
</traceback>
"""

    response = requests.post(
        "https://aipipe.org/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Python debugging assistant. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=30,
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    analysis = ErrorAnalysis.model_validate_json(content)

    total_lines = len(code.splitlines())

    # Remove invalid line numbers and duplicates.
    valid_lines = sorted(
        {
            line
            for line in analysis.error_lines
            if 1 <= line <= total_lines
        }
    )

    return valid_lines


@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    # AI is called only when execution fails.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    error_traceback = execution["output"]

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            error_traceback,
        )
    except Exception:
        # Fallback: extract the submitted-code line from the traceback.
        matches = re.findall(
            r'File "<string>", line (\d+)',
            error_traceback,
        )

        total_lines = len(request.code.splitlines())

        error_lines = (
            [int(matches[-1])]
            if matches and 1 <= int(matches[-1]) <= total_lines
            else []
        )

    return {
        "error": error_lines,
        "result": error_traceback,
    }
