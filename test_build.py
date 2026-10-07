"""Smoke tests for one converted AREX-2 MLX build.

Usage: venv/bin/python test_build.py builds/AREX-2-4bit
Writes results/<build>.json. Checks: plain answer, tool-call round trip, thinking
on/off, long-prompt recall, image input. Records speeds and peak memory.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load
from PIL import Image, ImageDraw, ImageFont

build = Path(sys.argv[1])
out_dir = Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)

t0 = time.time()
model, processor = load(str(build))
load_s = time.time() - t0
tok = processor.tokenizer
results = {"build": build.name, "load_seconds": round(load_s, 1), "tests": {}}


def chat(messages, max_tokens=512, image=None, **tpl):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **tpl)
    r = generate(model, processor, prompt, image=image, max_tokens=max_tokens, temperature=0.0)
    return r


def stats(r):
    return {
        "prompt_tokens": getattr(r, "prompt_tokens", None),
        "generation_tokens": getattr(r, "generation_tokens", None),
        "prompt_tps": round(getattr(r, "prompt_tps", 0) or 0, 1),
        "generation_tps": round(getattr(r, "generation_tps", 0) or 0, 1),
        "peak_memory_gb": round(getattr(r, "peak_memory", 0) or 0, 1),
    }


def record(name, passed, r, note=""):
    results["tests"][name] = {"pass": bool(passed), "note": note, "output": r.text[-600:], **stats(r)}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {stats(r)}  {note}", flush=True)


# 1. Plain answer, thinking off
r = chat([{"role": "user", "content": "What is 17 times 23? Reply with the number only."}],
         max_tokens=32, enable_thinking=False)
record("plain_answer", "391" in r.text and "<think>" not in r.text, r)

# 2. Thinking on: reasoning block closes and the answer is right
r = chat([{"role": "user", "content": "A train leaves at 14:35 and arrives at 17:10 the same day. "
           "How many minutes is the trip? End with 'ANSWER: <number>'."}],
         max_tokens=1500, enable_thinking=True)
m = re.search(r"ANSWER:\s*(\d+)", r.text.split("</think>")[-1])
record("thinking_on", "</think>" in r.text and m is not None and m.group(1) == "155", r,
       note=f"thinking_closed={'</think>' in r.text}")

# 3. Tool-call round trip
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Lisbon right now? Use the tool."}]
r = chat(msgs, max_tokens=200, tools=tools, enable_thinking=False)
# The chat template's format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
call = re.search(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", r.text, re.S)
parsed, ok = None, False
if call:
    args = {k: v.strip() for k, v in re.findall(r"<parameter=([\w.-]+)>(.*?)</parameter>", call.group(2), re.S)}
    parsed = {"name": call.group(1), "arguments": args}
    ok = parsed["name"] == "get_weather" and args.get("city", "").lower() == "lisbon"
record("tool_call_emitted", ok, r, note=f"parsed={parsed}")
if ok:
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": parsed}]},
             {"role": "tool", "content": json.dumps({"city": "Lisbon", "temp_c": 18, "sky": "light rain"})}]
    r = chat(msgs, max_tokens=200, tools=tools, enable_thinking=False)
    record("tool_result_used", "18" in r.text and "rain" in r.text.lower(), r)

# 4. Long prompt: one planted fact in about 8k tokens of filler
random.seed(7)
words = "ledger harbor window copper meadow signal paper orbit timber candle river anchor".split()
paras = [" ".join(random.choice(words) for _ in range(90)) + "." for _ in range(70)]
paras.insert(41, "Note for staff: the access code for the Lisbon office storeroom is 7341.")
r = chat([{"role": "user", "content": "\n\n".join(paras) +
           "\n\nWhat is the access code for the Lisbon office storeroom? Reply with the number only."}],
         max_tokens=32, enable_thinking=False)
record("long_prompt_recall", "7341" in r.text, r)

# 5. Image input: text and colour in a generated picture
img_path = out_dir / "test-image.png"
if not img_path.exists():
    im = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((60, 90, 260, 290), fill=(220, 30, 30))
    d.rectangle((380, 110, 560, 290), fill=(30, 60, 220))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
    except OSError:
        font = ImageFont.load_default()
    d.text((200, 320), "MANGO 47", fill="black", font=font)
    im.save(img_path)
prompt = apply_chat_template(processor, model.config,
                             "What text is written in this image, and what colour is the circle?",
                             num_images=1, enable_thinking=False)
r = generate(model, processor, prompt, image=[str(img_path)], max_tokens=120, temperature=0.0)
low = r.text.lower()
record("image_input", "mango" in low and "47" in low and "red" in low, r)

# 6. Writing speed on a longer answer
r = chat([{"role": "user", "content": "Write a Python function that merges overlapping intervals, "
           "with a short docstring and three example calls."}], max_tokens=400, enable_thinking=False)
record("write_speed", (getattr(r, "generation_tokens", 0) or 0) > 100, r)

results["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 1)
results["all_pass"] = all(t["pass"] for t in results["tests"].values())
(out_dir / f"{build.name}.json").write_text(json.dumps(results, indent=2))
print("ALL PASS" if results["all_pass"] else "SOME FAILED", "| peak GB", results["peak_memory_gb"],
      "| load s", results["load_seconds"])
