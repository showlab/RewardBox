"""LRM prompt/parse contract from pinned server 5845ab0.

The opt-in vLLM executor uses these messages instead of the Flask transport.
Offline parity tests execute the original server functions with a recorder.
"""
from __future__ import annotations

import json
import re


def messages_for(head: str, task: str, images: list) -> list[dict]:
    if head == "completion":
        prompt = (
            "Task: Whether the task has been completed.\n\n"
            f"The task is: {task}\n\n"
            "You are given:\n"
            "- Current observation: <image>\n\n"
            "Has the task been completed?\n"
            "Output your answer in the following JSON format:\n"
            '{ "task_completed": "yes" or "no" }'
        )
    elif head == "contrastive":
        prompt = (
            "Task: Compare the completion progress.\n\n"
            f"The task is: {task}.\n\n"
            "You are given:\n"
            "- Later observation (Image A): <image>\n"
            "- Later observation (Image B): <image>\n\n"
            "Question: Which of Image A or Image B is closer to completing the task?\n"
            "Select one value from the following list:\n"
            '["ImageA", "ImageB"]\n\n'
            "Please provide a step-by-step visual analysis first, and then output your answer "
            "in the following JSON format:\n"
            '{ "more_complete_image": "selected_value" }'
        )
    else:
        raise ValueError(f"Unsupported point/pair head: {head}")
    return _inline(prompt, images, strip=head == "completion")


def progress_messages(task: str, current, *, initial=None, goal=None) -> list[dict]:
    images = []
    parts = ["Task: Estimate the completion progress.", f"\nThe task is: {task}",
             "\nYou are given:"]
    if goal is not None:
        parts.append("\n- Goal Standard: <image>")
        images.append(goal)
    if initial is not None:
        parts.append("\n- Initial observation: <image>")
        images.append(initial)
    parts.append("\n- Current observation: <image>")
    images.append(current)
    if goal is not None:
        parts.append("""

Question: Compare the Current Observation with the Goal Standard. Estimate the completion progress as a value between 0.0 and 1.0.
Select one value from the following list:
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Definitions:
- 0.0: Task has just started.
- 0.1 - 0.9: Intermediate progress steps (assess visual distance to Goal).
- 1.0: Task is Finished.
Output your answer in the following JSON format:
{ "completion_progress": selected_value }""")
    else:
        parts.append("""

Question: Based on the task description, estimate the completion progress as a value between 0.0 and 1.0.
Select one value from the following list:
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Definitions:
- 0.0: Task has just started.
- 0.1 - 0.9: Intermediate progress steps (assess how close to completion).
- 1.0: Task is Finished.
Output your answer in the following JSON format:
{ "completion_progress": selected_value }""")
    return _inline("".join(parts), images)


def _inline(prompt: str, images: list, *, strip: bool = False) -> list[dict]:
    content = []
    for index, segment in enumerate(prompt.split("<image>")):
        if segment.strip():
            content.append({"type": "text", "text": segment.strip() if strip else segment})
        if index < len(images):
            content.append({"type": "image", "image": images[index]})
    return [{"role": "user", "content": content}]


def parse_completion(response: str) -> bool:
    matches = re.findall(r'"task_completed"\s*:\s*"(yes|no)"', response, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == "yes"
    lower = response.lower()
    return "yes" in lower and "no" not in lower


def parse_comparison(response: str) -> str | None:
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group(0))
            if "more_complete_image" in data:
                value = data["more_complete_image"]
                if value in ("ImageA", "ImageB"):
                    return value
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    match = re.search(r'"more_complete_image"\s*:\s*"(Image[AB])"', response)
    if match:
        return match.group(1)
    last_a, last_b = response.rfind("ImageA"), response.rfind("ImageB")
    if last_b > last_a:
        return "ImageB"
    if last_a > last_b:
        return "ImageA"
    return None


def parse_progress(response: str) -> float:
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group(0))
            if "completion_progress" in data:
                value = data["completion_progress"]
                if value in ("NA", "na"):
                    return 0.0
                return max(0.0, min(1.0, float(value)))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        match = re.search(r'"completion_progress"\s*:\s*"?(NA|([0-9.]+))"?',
                          response, re.IGNORECASE)
        if match:
            captured = match.group(1)
            return 0.0 if captured.upper() == "NA" else max(0.0, min(1.0, float(captured)))
    except (ValueError, TypeError):
        pass
    for pattern in (r"\b(0(\.[0-9]+)?|1(\.0)?)\b", r"Progress[:\s]*([0-9]*\.?[0-9]+)",
                    r"progress[:\s]*([0-9]*\.?[0-9]+)", r"Score[:\s]*([0-9]*\.?[0-9]+)",
                    r"score[:\s]*([0-9]*\.?[0-9]+)", r"([0-9]*\.?[0-9]+)\s*$",
                    r"^([0-9]*\.?[0-9]+)"):
        match = re.search(pattern, response)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                continue
    return 0.0
