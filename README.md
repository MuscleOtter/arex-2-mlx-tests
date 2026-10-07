# AREX-2 on Mac: MLX builds, tests and results

I converted [BAAI/AREX-2](https://huggingface.co/BAAI/AREX-2) (a Qwen3.8-27B fine-tune for coding and agent work, with image input) to MLX for Apple Silicon, and tested the builds before publishing them. This repo holds the test scripts and the raw results behind the numbers on the model pages.

**The builds:** [4-bit](https://huggingface.co/mlx-community/AREX-2-4bit) · [5-bit](https://huggingface.co/mlx-community/AREX-2-5bit) · [6-bit](https://huggingface.co/mlx-community/AREX-2-6bit) · [8-bit](https://huggingface.co/mlx-community/AREX-2-8bit) · [all four as a collection](https://huggingface.co/collections/muscleotter/arex-2-for-mac-mlx-6ac6547977445e4344948ae7)

## What I found

- **The 5-bit, 6-bit and 8-bit are too close to tell apart** on these tests. The 5-bit needs the least memory of the three.
- **The 4-bit is the weakest size.** It loops at temperature 0 and needs about twice the room to think. Seven different 4-bit recipes all landed between 88% and 93% agreement with the original.
- **AREX-2 was more efficient than plain Qwen3.8-27B:** about 46% fewer tokens and about half the time on the same tests, with thinking on. With thinking off, in an agent harness, the two were level.
- **A draft model made for plain Qwen3.8-27B works with AREX-2 unchanged,** taking the 8-bit from 8 to 19 tokens per second.

Everything was measured on one Mac mini M4 Pro with 64 GB. These are small tests, not benchmarks, and they say nothing about the long multi-round tasks AREX-2 was trained for.

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

## The 4-bit recipes

| Recipe | Same next token as the original |
|---|---|
| Plain 4-bit | 91.4% |
| Group size 32 | 92.6% |
| mixed_4_6 | 91.8% |
| mixed_4_8 | 92.0% |
| AWQ | 91.5% |
| nvfp4 | 89.8% |
| mxfp4 | 88.5% |

At temperature 0 the plain 4-bit got stuck repeating `</think>` on 4 of 18 easy tasks. With the recommended sampling it did not.

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

`<model>` is a local folder or a Hugging Face repo name such as `mlx-community/AREX-2-5bit`. Results are written to a `results` folder next to the scripts.

Two things to know before running them:

- **`quality.py` and `rounds.py` execute the code the model writes,** in a separate Python process with a time limit, to run the hidden tests. Run them somewhere you are comfortable with that.
- **The next-token comparison** reads the original model card from a local `source` folder as part of its fixed text. Point it at a copy of BAAI's README to reproduce the exact figure.

## Limits

- One Mac, with 64 GB. Nothing was run on a smaller machine.
- Small test sets, one or two runs each. The problems are well known, so both models have probably seen them in training.
- Short single tasks only. BAAI's own results are on long multi-round benchmarks, which these tests do not touch.

## Credit and license

The model is BAAI's ([paper](https://arxiv.org/abs/2609.38288)), built on Qwen3.8-27B, and released under Apache 2.0. The conversions were made with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.4. The scripts in this repo are under the MIT license.
