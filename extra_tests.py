"""Extra checks for one AREX-2 MLX build: long-prompt recall and a multi-step tool loop.

Usage:
  venv/bin/python extra_tests.py long  builds/AREX-2-5bit [--tokens 32000]
  venv/bin/python extra_tests.py agent builds/AREX-2-5bit [--label NAME]
Results go to results/long-<name>.json and results/agent-<name>.json.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load

ROOT = Path(__file__).parent
RES = ROOT / "results"
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def long_prompt(build, name, target):
    """Three facts planted at 15%, 50% and 85% of a long filler prompt; the model must return all three."""
    model, processor = load(str(build))
    tok = processor.tokenizer
    random.seed(21)
    words = "ledger harbor window copper meadow signal paper orbit timber candle river anchor".split()
    para = lambda: " ".join(random.choice(words) for _ in range(90)) + "."
    n = max(10, target // 101)   # one filler paragraph is about 101 tokens
    paras = [para() for _ in range(n)]
    facts = {"Lisbon": "7341", "Osaka": "2086", "Tulsa": "9517"}
    for (city, code), frac in zip(facts.items(), (0.15, 0.50, 0.85)):
        paras.insert(int(n * frac), f"Note for staff: the storeroom access code for the {city} office is {code}.")
    q = ("\n\nList the storeroom access codes for the Lisbon, Osaka and Tulsa offices, "
         "in that order, as three numbers separated by commas. Reply with the numbers only.")
    prompt = tok.apply_chat_template([{"role": "user", "content": "\n\n".join(paras) + q}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t0 = time.time()
    r = generate(model, processor, prompt, max_tokens=40, temperature=0.0)
    out = {"build": name, "prompt_tokens": r.prompt_tokens, "prompt_tps": round(r.prompt_tps, 1),
           "seconds": round(time.time() - t0, 1), "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 1),
           "found": {c: (code in r.text) for c, code in facts.items()}, "output": r.text[:200]}
    out["pass"] = all(out["found"].values())
    (RES / f"long-{name}-{target}.json").write_text(json.dumps(out, indent=2))
    print(f"{'PASS' if out['pass'] else 'FAIL'} long prompt {name}: {out['prompt_tokens']} tokens, "
          f"{out['prompt_tps']} tok/s, {out['seconds']}s, peak {out['peak_memory_gb']} GB, found {out['found']}", flush=True)


TOOLS = [
    {"type": "function", "function": {"name": "find_customer", "description": "Look up a customer by full name.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_orders", "description": "List a customer's orders with totals.",
     "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}}},
    {"type": "function", "function": {"name": "get_order", "description": "Get one order's items and tracking number.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "track_shipment", "description": "Get shipment status by tracking number.",
     "parameters": {"type": "object", "properties": {"tracking_number": {"type": "string"}}, "required": ["tracking_number"]}}},
]
ORDERS = [{"order_id": "A-1009", "date": "2026-08-02", "total": 129.50},
          {"order_id": "A-1042", "date": "2026-09-17", "total": 342.00},
          {"order_id": "A-1077", "date": "2026-09-29", "total": 58.25}]
TRACKING = {"A-1009": "TRK-70312", "A-1042": "TRK-88213", "A-1077": "TRK-90455"}


def run_tool(name, a):
    if name == "find_customer":
        return {"customer_id": "C-4471"} if "whitfield" in str(a.get("name", "")).lower() else {"error": "no such customer"}
    if name == "list_orders":
        return {"orders": ORDERS} if a.get("customer_id") == "C-4471" else {"error": "unknown customer_id"}
    if name == "get_order":
        oid = a.get("order_id")
        return {"order_id": oid, "items": 3, "tracking_number": TRACKING[oid]} if oid in TRACKING else {"error": "unknown order_id"}
    if name == "track_shipment":
        t = a.get("tracking_number")
        if t == "TRK-88213":
            return {"status": "delayed", "location": "Memphis", "eta": "2026-10-09"}
        return {"status": "delivered"} if t in TRACKING.values() else {"error": "unknown tracking_number"}
    return {"error": f"unknown tool {name}"}


def agent(build, name):
    """Four chained tool calls plus a sum. Pass needs the right shipment tracked and all three facts in the answer."""
    model, processor = load(str(build))
    tok = processor.tokenizer
    runs = []
    for seed in (31, 32, 33):
        msgs = [{"role": "user", "content": "Dana Whitfield wants to know where her most expensive order is and "
                 "when it will arrive. Also tell her the total she has spent across all her orders. Use the tools."}]
        calls, final, t0, toks = [], "", time.time(), 0
        for turn in range(8):
            prompt = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
            r = generate(model, processor, prompt, max_tokens=1500, seed=seed + turn, **SAMPLING)
            toks += r.generation_tokens
            text = r.text.split("</think>")[-1]
            found = re.findall(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", text, re.S)
            if not found:
                final = text.strip()
                break
            tcs = []
            for fn, body in found:
                a = {k: v.strip() for k, v in re.findall(r"<parameter=([\w.-]+)>(.*?)</parameter>", body, re.S)}
                calls.append((fn, a))
                tcs.append({"type": "function", "function": {"name": fn, "arguments": a}})
            msgs.append({"role": "assistant", "content": text.split("<tool_call>")[0].strip(), "tool_calls": tcs})
            for tc in tcs:
                msgs.append({"role": "tool", "content": json.dumps(run_tool(tc["function"]["name"], tc["function"]["arguments"]))})
        low = final.lower()
        ok = (("track_shipment", {"tracking_number": "TRK-88213"}) in calls and "memphis" in low and "529.75" in final
              and any(d in low for d in ("2026-10-09", "october 9", "oct 9", "oct. 9", "9 october", "10/09", "10/9")))
        runs.append({"seed": seed, "pass": ok, "tool_calls": [c[0] for c in calls], "turns": turn + 1, "tokens": toks,
                     "seconds": round(time.time() - t0, 1), "final": final[-500:]})
        print(f"{'PASS' if ok else 'FAIL'} agent {name} seed {seed}: {len(calls)} calls, {toks} tok, {time.time() - t0:.0f}s", flush=True)
    out = {"build": name, "passed": sum(x["pass"] for x in runs), "runs": runs}
    (RES / f"agent-{name}.json").write_text(json.dumps(out, indent=2))
    print(f"agent {name}: {out['passed']} of {len(runs)} runs passed", flush=True)


if __name__ == "__main__":
    mode, build = sys.argv[1], Path(sys.argv[2])
    name = arg("--label", build.name)
    RES.mkdir(exist_ok=True)
    if mode == "long":
        long_prompt(build, name, int(arg("--tokens", "32000")))
    else:
        agent(build, name)
