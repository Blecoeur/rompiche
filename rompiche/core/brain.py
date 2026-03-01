import os
import mistralai
from mistralai.models.usermessage import UserMessage
from mistralai.models.systemmessage import SystemMessage
import json

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal


class MismatchExplanationResponse(BaseModel):
    issue_summary: str = Field(
        description="One-sentence summary of what went wrong in this prediction.",
        default="",
    )
    likely_root_cause: str = Field(
        description="Root cause of the mismatch (e.g. wrong format, missing field, ambiguous instruction).",
        default="",
    )
    format_rules_violated: List[str] = Field(
        description="Specific formatting or extraction rules that were violated.",
        default_factory=list,
    )
    fix_suggestion: str = Field(
        description="Concrete suggestion for how the prompt or schema could be changed to avoid this error.",
        default="",
    )


class BrainResponse(BaseModel):
    decision: Literal["stop", "continue"] = Field(description="The decision to make", default="continue")
    reason: str = Field(description="The reason for the decision", default="")
    changes: list[str] = Field(
        description="Concrete list of prompt/schema changes made in this decision.",
        default_factory=list,
    )
    updated_prompt: str | None = Field(
        description="The full updated prompt text, or null if no prompt change.",
        default=None,
    )
    updated_schema: dict | None = Field(
        description="The full updated schema object, or null if no schema change.",
        default=None,
    )


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


def _track_explainer_tokens(response) -> None:
    """Track token usage from API response on explain_mismatch."""
    tokens_used = 0
    if hasattr(response, "usage") and response.usage:
        if hasattr(response.usage, "total_tokens"):
            tokens_used = response.usage.total_tokens
        elif hasattr(response.usage, "prompt_tokens") and hasattr(
            response.usage, "completion_tokens"
        ):
            tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

    if "tokens_used" not in explain_mismatch.__dict__:
        explain_mismatch.tokens_used = 0
    explain_mismatch.tokens_used += tokens_used


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
    metrics, prompt, schema, evaluator_config, mismatch_examples=None, hints=None, performance_worsened=False
):
    """
    Asks the AI model if the results meet the success thresholds.
    Returns: {
        "decision": "stop/continue",
        "reason": "...",
        "changes": ["...", "..."],
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
- Provide a "changes" array containing concise, concrete change items.
- Each item in "changes" must describe one specific modification made to prompt/schema.
- Use null for "updated_prompt" and/or "updated_schema" when that part is unchanged.

VERY IMPORTANT:
- DO NOT HARDCODE THE VALUES IN THE PROMPT OR SCHEMA, YOU CAN GIVE EXAMPLES BUT INVENT SIMILAR VALUES TO THE GROUND TRUTH.
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

Analyze the differences carefully and fix the prompt/schema so predictions match ground truth exactly.

Don't forget to follow the instructions above."""

    if hints:
        hints_str = "\n".join([f"- {hint}" for hint in hints])
        user_prompt += f"""

USER HINTS:
{hints_str}

Consider these hints when improving the prompt and schema."""

    if performance_worsened:
        user_prompt += """

WARNING: Performance worsened compared to previous iteration. Be very conservative with changes. Focus on reverting to previous working configuration or making minimal, targeted fixes only."""

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
            '  \"changes\": [\"specific change 1\", \"specific change 2\"],\n'
            '  \"updated_prompt\": \"the full improved prompt text\" or null,\n'
            '  \"updated_schema\": { \"... the full improved schema object ...\" } or null\n'
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


def _normalize_explanation_response(parsed) -> Dict[str, Any]:
    """Normalize/validate a MismatchExplanationResponse to a JSON-serializable dict."""
    if isinstance(parsed, BaseModel):
        data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
    elif isinstance(parsed, dict):
        data = parsed
    else:
        data = dict(parsed)

    validated = MismatchExplanationResponse(**{
        k: v for k, v in data.items()
        if k in MismatchExplanationResponse.model_fields
    })
    return validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()


def explain_mismatch(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Send a mismatch to the LLM with contextual messages built by the processor
    and return a structured explanation.

    Args:
        messages: A list of OpenAI-style message dicts (role/content). The caller
                  (processor hook or fallback builder) is responsible for injecting
                  the right modalities (text, image_url, etc.).

    Returns:
        A dict matching MismatchExplanationResponse fields:
        {
            "issue_summary": str,
            "likely_root_cause": str,
            "format_rules_violated": [str, ...],
            "fix_suggestion": str,
        }
    """
    client = mistralai.Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    system_message = {
        "role": "system",
        "content": (
            "You are an expert extraction evaluator. You will be shown an extraction mismatch: "
            "the original input, what the model predicted, and what the correct answer is. "
            "Your job is to explain what went wrong and why.\n\n"
            "Be specific and concise. Focus on the exact difference between prediction and ground truth, "
            "and identify concrete formatting or instruction issues that caused the error.\n\n"
            "Respond with a JSON object using this exact structure:\n"
            "{\n"
            '  "issue_summary": "one sentence describing what was wrong",\n'
            '  "likely_root_cause": "the underlying cause (e.g. missing instruction, wrong format assumed)",\n'
            '  "format_rules_violated": ["rule 1 that was broken", "rule 2 ..."],\n'
            '  "fix_suggestion": "specific prompt/schema change that would fix this"\n'
            "}"
        ),
    }

    full_messages = [system_message] + list(messages)

    try:
        response = client.chat.parse(
            model="mistral-large-latest",
            messages=full_messages,
            temperature=0.2,
            response_format=MismatchExplanationResponse,
        )
        _track_explainer_tokens(response)
        parsed = response.choices[0].message.parsed
        return _normalize_explanation_response(parsed)
    except Exception:
        fallback_system = (
            "You are an expert extraction evaluator. Explain what went wrong in this mismatch. "
            "Respond ONLY in valid JSON:\n"
            "{\n"
            '  "issue_summary": "...",\n'
            '  "likely_root_cause": "...",\n'
            '  "format_rules_violated": ["..."],\n'
            '  "fix_suggestion": "..."\n'
            "}"
        )
        fallback_messages = [{"role": "system", "content": fallback_system}] + list(messages)
        try:
            fallback_response = client.chat.complete(
                model="mistral-large-latest",
                messages=fallback_messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            _track_explainer_tokens(fallback_response)
            raw_content = fallback_response.choices[0].message.content
            parsed_fallback = _extract_json_from_content(raw_content)
            return _normalize_explanation_response(parsed_fallback)
        except Exception as e:
            return {
                "issue_summary": f"Explanation failed: {e}",
                "likely_root_cause": "",
                "format_rules_violated": [],
                "fix_suggestion": "",
            }

