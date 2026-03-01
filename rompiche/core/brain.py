import os
import mistralai
from mistralai.models.usermessage import UserMessage
from mistralai.models.systemmessage import SystemMessage
import json

from pydantic import BaseModel, Field
from typing import Literal

class BrainResponse(BaseModel):
    decision: Literal["stop", "continue"] = Field(description="The decision to make", default="continue")
    reason: str = Field(description="The reason for the decision", default="")
    updated_prompt: str = Field(description="The updated prompt", default="")
    updated_schema: dict = Field(description="The updated schema", default_factory=dict)


def _track_brain_tokens(response) -> None:
    """Track token usage from API response on the function object."""
    tokens_used = 0
    if hasattr(response, "usage") and response.usage:
        if hasattr(response.usage, "total_tokens"):
            tokens_used = response.usage.total_tokens
        elif hasattr(response.usage, "prompt_tokens") and hasattr(
            response.usage, "completion_tokens"
        ):
            tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

    if "tokens_used" not in get_brain_decision.__dict__:
        get_brain_decision.tokens_used = 0
    get_brain_decision.tokens_used += tokens_used


def _normalize_brain_response(parsed) -> dict:
    """Normalize/validate response to a JSON-serializable BrainResponse dict."""
    if isinstance(parsed, BaseModel):
        data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
    elif isinstance(parsed, dict):
        data = parsed
    else:
        data = dict(parsed)

    validated = BrainResponse(**data)
    return validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()


def _extract_json_from_content(content) -> dict:
    """Extract a JSON object from model content that may include markdown fencing."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


def get_brain_decision(
    metrics, prompt, schema, evaluator_config, mismatch_examples=None, hints=None
):
    """
    Asks the AI model if the results meet the success thresholds.
    Returns: {
        "decision": "stop/continue",
        "reason": "...",
        "updated_prompt": "...",
        "updated_schema": {...}
    }
    """
    client = mistralai.Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    system_prompt = """You are the optimization brain. You analyze extraction results and improve prompts/schemas.

Your job:
1. Check if metrics meet the success criteria.
2. If not, analyze the mismatch examples to understand WHY predictions differ from ground truth.
3. Produce an improved prompt and schema that address the root causes.

IMPORTANT:
- Study the mismatch examples carefully. Look at exact formatting differences (e.g. time formats, casing, wording).
- The updated prompt and schema must produce output that matches the ground truth format EXACTLY.
- Do NOT just add vague instructions — be specific about expected formats based on what the ground truth shows.
"""

    user_prompt = f"""Metrics: {json.dumps(metrics)}
Success thresholds: {json.dumps(evaluator_config.get("success_thresholds", {}))}
Current prompt: {prompt}
Current schema: {json.dumps(schema)}"""

    if mismatch_examples:
        examples_str = json.dumps(mismatch_examples, indent=2)
        user_prompt += f"""

Here are concrete mismatch examples (prediction vs ground_truth):
{examples_str}

Analyze the differences carefully and fix the prompt/schema so predictions match ground truth exactly."""

    if hints:
        hints_str = "\n".join([f"- {hint}" for hint in hints])
        user_prompt += f"""

USER HINTS:
{hints_str}

Consider these hints when improving the prompt and schema."""

    messages = [SystemMessage(content=system_prompt), UserMessage(content=user_prompt)]

    try:
        response = client.chat.parse(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.3,
            response_format=BrainResponse,
        )
        _track_brain_tokens(response)
        parsed = response.choices[0].message.parsed
        return _normalize_brain_response(parsed)
    except Exception:
        # Fallback path when structured parse returns malformed JSON.
        fallback_system_prompt = (
            f"{system_prompt}\n\n"
            "Respond ONLY in valid JSON using this structure:\n"
            "{\n"
            '  \"decision\": \"stop\" or \"continue\",\n'
            '  \"reason\": \"...\",\n'
            '  \"updated_prompt\": \"the full improved prompt text\",\n'
            '  \"updated_schema\": { \"... the full improved schema object ...\" }\n'
            "}"
            "Do not include any extra commentary, markdown, or non-JSON output."
        )

        fallback_messages = [
            {"role": "system", "content": fallback_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        fallback_response = client.chat.complete(
            model="mistral-large-latest",
            messages=fallback_messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        _track_brain_tokens(fallback_response)
        raw_content = fallback_response.choices[0].message.content
        parsed_fallback = _extract_json_from_content(raw_content)
        return _normalize_brain_response(parsed_fallback)

