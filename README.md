# AREX-2 on Mac (MLX): tested builds, speed and quality results

**AREX-2 runs well on Apple Silicon Macs with 32 GB of memory or more.** This repo has the numbers behind that claim and the scripts to check them yourself.

[AREX-2](https://huggingface.co/BAAI/AREX-2) is BAAI's open 27B model for coding, reasoning and agent work, built on Qwen3.8-27B, with image input. I converted it to MLX, the format that runs natively on Macs, in four sizes, and tested every one before publishing. As far as I can find, this is also the only published side-by-side of AREX-2 and the model it was built from.

**Get the builds:** [4-bit](https://huggingface.co/mlx-community/AREX-2-4bit) · [5-bit](https://huggingface.co/mlx-community/AREX-2-5bit) · [6-bit](https://huggingface.co/mlx-community/AREX-2-6bit) · [8-bit](https://huggingface.co/mlx-community/AREX-2-8bit) · [all four as a collection](https://huggingface.co/collections/muscleotter/arex-2-for-mac-mlx-6ac6547977445e4344948ae7)

## Quick answers

**Can AREX-2 run on a Mac?**
Yes. The MLX builds run on Apple Silicon. They were tested on a Mac mini M4 Pro with 64 GB, and the 5-bit uses 20 to 28 GB, so it should fit a 32 GB Mac.

**Which AREX-2 quantization should I use on a Mac?**
The 5-bit, 6-bit and 8-bit scored too close to tell apart, so take the largest that fits your memory. Use the 4-bit only if the 5-bit does not fit.

**How fast is AREX-2 on an M4 Pro?**
8 to 14 tokens per second depending on size, and 19 to 27 with a draft model. Reading a prompt runs at about 90 tokens per second.

**Is AREX-2 better than Qwen3.8-27B?**
In these tests it was more efficient: about 46% fewer tokens and about half the time, with a few more problems solved. With thinking off, in an agent harness, the two were level.

**Does AREX-2 work in LM Studio?**
Yes, including images and tool calls. Paste `https://huggingface.co/mlx-community/AREX-2-5bit` into LM Studio's model search.

**Does it work in Ollama?**
Not these builds. Ollama uses the GGUF format, so use a GGUF version of AREX-2 there.

**How do I make AREX-2 faster on a Mac?**
Use a draft model. The DFlash2 helper made for plain Qwen3.8-27B works with AREX-2 unchanged and took the 8-bit from 8 to 19 tokens per second.

**Does it work in agent harnesses?**
It was tested in MiniMax Code, where the 8-bit fixed three small failing projects in 6 of 6 trials. Other harnesses should work through LM Studio's local server, which returns standard tool calls.

## Run it in two minutes

Without coding: install [LM Studio](https://lmstudio.ai), paste `https://huggingface.co/mlx-community/AREX-2-5bit` into its model search, download, and chat.

From the command line:

```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate \
  --model mlx-community/AREX-2-5bit \
  --max-tokens 4000 --temperature 1.0 --top-p 0.95 --top-k 20 \
  --prompt "Write a Python function that merges overlapping intervals."
```

Keep the sampling at temperature 1.0, top-p 0.95, top-k 20, and give it a response limit of 4,000 tokens or more, because it thinks before it answers.

## The sizes

| Size | Download | Memory used | Writing speed | With a draft model | Same next token as the original | Hard problems, first try | Within 3 tries |
|---|---|---|---|---|---|---|---|
| 4-bit | 16.1 GB | 16 to 24 GB | 14 tok/s | 27 tok/s | 91.4% | 3 of 12 | 9 of 12 |
| 5-bit | 19.4 GB | 20 to 28 GB | 12 tok/s | 20 tok/s | 95.9% | 8 of 12 | 11 of 12 |
| 6-bit | 22.8 GB | 23 to 31 GB | 10 tok/s | 19 tok/s | 97.6% | 9 of 12 | 10 of 12 |
| 8-bit | 29.5 GB | 30 to 38 GB | 8 tok/s | 19 tok/s | 99.1% | 7 of 12 | 12 of 12 |

- **Memory used** runs from a short chat to a 32,000-token prompt.
- **Same next token as the original** is how often the build picks the same next token as the full-precision weights, over a fixed 6,141-token text.
- **Hard problems** are 12 Python problems with a 4,000-token limit, where the model sees its failing test and gets up to 3 tries. One run per size, so a gap of one or two is noise.
- **With an 8,000-token limit** the 4-bit solved 8 of 12 on the first try and the 5-bit solved 11 of 12.
- **Draft model:** [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) through mlx-dspark 0.20.1.

## AREX-2 against plain Qwen3.8-27B

Both at 8-bit MLX, with the same prompts, sampling (temperature 1.0, top-p 0.95, top-k 20), seeds and a 4,000-token limit. Two runs of each test, thinking on.

| | AREX-2 | Qwen3.8-27B |
|---|---|---|
| Tokens used, all tests | 113,795 | 212,514 |
| Time, all tests | 249 min | 451 min |
| Easy tasks passed | 35 of 36 | 30 of 36 |
| Hard problems, first try | 15 of 24 | 11 of 24 |
| Hard problems, within 3 tries | 23 of 24 | 20 of 24 |

Every failed attempt by plain Qwen3.8-27B ran into the token limit (31 of 31), against 9 of 13 for AREX-2. So most of the gap is how long each model thinks, and with a higher limit the plain model might solve as many, more slowly.

**In an agent harness, with thinking off, they were level.** In MiniMax Code 0.5.9, each model fixed three small failing projects in 6 of 6 trials, in 17 and 18 minutes.

## Why the 4-bit is the weak one

| 4-bit recipe | Same next token as the original |
|---|---|
| Plain | 91.4% |
| Group size 32 | 92.6% |
| mixed_4_6 | 91.8% |
| mixed_4_8 | 92.0% |
| AWQ | 91.5% |
| nvfp4 | 89.8% |
| mxfp4 | 88.5% |

None of seven recipes got close to the 5-bit's 95.9%. At temperature 0 the plain 4-bit also got stuck repeating `</think>` on 4 of 18 easy tasks; with the recommended sampling it did not. It needs about twice the room to think, so give it a response limit of 8,000 tokens or more.

## What is in this repo

| File | What it does |
|---|---|
| `test_build.py` | Smoke tests: plain answer, thinking, one tool call, recall in a 7,000-token prompt, one image. |
| `quality.py` | Next-token comparison on a fixed text, and 18 easy coding tasks with hidden tests. |
| `rounds.py` | 12 hard problems, with the failing test shown to the model for up to 3 tries. |
| `extra_tests.py` | Recall in a 32,000-token prompt, and a four-step tool loop. |
| `image_tests.py` | 11 image questions. The images are drawn by the script, so every answer is known. |
| `harness_trials.py` | MiniMax Code fixes three small failing projects, run headless against a local server. |
| `results/` | The raw output behind every number above, including the answers the models gave. |

## Running the tests

```bash
pip install -U mlx-vlm
python quality.py --selfcheck                      # checks the task tests against reference solutions
python quality.py <model> --sample                 # 18 easy tasks, recommended sampling
python rounds.py <model> --label NAME --seed 41    # 12 hard problems
python extra_tests.py long <model>
python extra_tests.py agent <model>
python image_tests.py <model>
python test_build.py <model>
```

`<model>` is a local folder or a Hugging Face repo name such as `mlx-community/AREX-2-5bit`. The scripts work on any MLX model that mlx-vlm can load, so you can reuse them to test other conversions.

Two things to know before running them:

- **`quality.py` and `rounds.py` execute the code the model writes,** in a separate Python process with a time limit, to run the hidden tests. Run them somewhere you are comfortable with that.
- **The next-token comparison** reads the original model card from a local `source` folder as part of its fixed text. Point it at a copy of BAAI's README to reproduce the exact figure.

## Limits

- One Mac, with 64 GB. Nothing was run on a smaller machine, so the advice for 32 GB is an estimate from measured memory use.
- Small test sets, one or two runs each. The problems are well known, so both models have probably seen them in training.
- Short single tasks only. BAAI's own results are on long multi-round benchmarks, which these tests do not touch.

## Questions

Open an issue here, or a discussion on any of the model pages.

## Credit and license

The model is BAAI's ([paper](https://arxiv.org/abs/2609.38288)), built on Qwen3.8-27B, and released under Apache 2.0. The conversions were made with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.4. The scripts in this repo are under the MIT license.
