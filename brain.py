import os
import mistralai
from mistralai.models.usermessage import UserMessage
from mistralai.models.systemmessage import SystemMessage
from mistralai.models.assistantmessage import AssistantMessage
import json

def get_brain_decision(metrics, prompt, schema, success_criteria, mismatch_examples=None):
    """
    Asks Mistral if the results meet the success criteria.
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

ALWAYS respond in JSON with this structure:
{
    "decision": "stop" or "continue",
    "reason": "...",
    "updated_prompt": "the full improved prompt text",
    "updated_schema": { ... the full improved schema object ... }
}"""

    user_prompt = f"""Metrics: {json.dumps(metrics)}
Success criteria: {json.dumps(success_criteria)}
Current prompt: {prompt}
Current schema: {json.dumps(schema)}"""

    if mismatch_examples:
        examples_str = json.dumps(mismatch_examples, indent=2)
        user_prompt += f"""

Here are concrete mismatch examples (prediction vs ground_truth):
{examples_str}

Analyze the differences carefully and fix the prompt/schema so predictions match ground truth exactly."""

    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=user_prompt)
    ]

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# Example usage
if __name__ == "__main__":
    metrics = {"date_exact_match": 0.85, "title_similarity": 0.60}
    success_criteria = {"date_exact_match": 1.0, "title_similarity": 0.8}
    prompt = "Extract the title and date from the text."
    schema = {"title": "str", "date": "YYYY-MM-DD"}

    decision = get_brain_decision(metrics, prompt, schema, success_criteria)
    print(json.dumps(decision, indent=2))