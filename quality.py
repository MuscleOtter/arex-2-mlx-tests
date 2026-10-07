"""Quality comparison across AREX-2 MLX builds.

Usage:
  venv/bin/python quality.py --selfcheck                 # verify the task tests against reference solutions
  venv/bin/python quality.py builds/AREX-2-4bit [--stop-at HH:MM]
  venv/bin/python quality.py --compare                   # table across builds, agreement against 8-bit

Two measures per build:
  1. Next-token loss on a fixed 6,144-token text (Python stdlib source + the model README), and
     the per-position top choice, so builds can be compared token by token against the 8-bit build.
  2. 18 coding tasks with hidden tests, default chat template settings (thinking on), temperature 0.
"""
import datetime
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

TASKS = [
    ("merge_intervals", "merge_intervals(intervals: list[list[int]]) -> list[list[int]]: merge all overlapping "
     "closed intervals (touching intervals like [1,2] and [2,3] merge) and return them sorted by start.",
     "def merge_intervals(intervals):\n    out=[]\n    for s,e in sorted(intervals):\n        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)\n        else: out.append([s,e])\n    return out",
     ["assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]",
      "assert merge_intervals([[1,2],[2,3]])==[[1,3]]", "assert merge_intervals([])==[]",
      "assert merge_intervals([[5,7],[1,10],[2,3]])==[[1,10]]"]),
    ("int_to_roman", "int_to_roman(n: int) -> str: convert 1..3999 to a Roman numeral.",
     "def int_to_roman(n):\n    v=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    s=''\n    for a,b in v:\n        while n>=a: s+=b; n-=a\n    return s",
     ["assert int_to_roman(1994)=='MCMXCIV'", "assert int_to_roman(3999)=='MMMCMXCIX'", "assert int_to_roman(4)=='IV'",
      "assert int_to_roman(58)=='LVIII'"]),
    ("longest_palindrome", "longest_palindrome(s: str) -> str: return the longest palindromic substring; "
     "if several have the same length return the one that starts first.",
     "def longest_palindrome(s):\n    best=''\n    for i in range(len(s)):\n        for l,r in ((i,i),(i,i+1)):\n            while l>=0 and r<len(s) and s[l]==s[r]: l-=1; r+=1\n            if r-l-1>len(best): best=s[l+1:r]\n    return best",
     ["assert longest_palindrome('babad')=='bab'", "assert longest_palindrome('cbbd')=='bb'",
      "assert longest_palindrome('')==''", "assert longest_palindrome('forgeeksskeegfor')=='geeksskeeg'",
      "assert longest_palindrome('abcd')=='a'"]),
    ("eval_rpn", "eval_rpn(tokens: list[str]) -> int: evaluate reverse Polish notation with + - * /; "
     "division truncates toward zero.",
     "def eval_rpn(tokens):\n    st=[]\n    for t in tokens:\n        if t in '+-*/' and len(t)==1:\n            b=st.pop(); a=st.pop()\n            st.append(a+b if t=='+' else a-b if t=='-' else a*b if t=='*' else int(a/b))\n        else: st.append(int(t))\n    return st[0]",
     ["assert eval_rpn(['2','1','+','3','*'])==9", "assert eval_rpn(['4','13','5','/','+'])==6",
      "assert eval_rpn(['7','-2','/'])==-3", "assert eval_rpn(['-7','2','/'])==-3",
      "assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+'])==22"]),
    ("spiral_order", "spiral_order(matrix: list[list[int]]) -> list[int]: return the elements in clockwise "
     "spiral order starting at the top-left. The matrix may be empty or non-square.",
     "def spiral_order(m):\n    out=[]\n    m=[r[:] for r in m]\n    while m and m[0]:\n        out+=m.pop(0)\n        m=[list(r) for r in zip(*m)][::-1]\n    return out",
     ["assert spiral_order([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]",
      "assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]])==[1,2,3,4,8,12,11,10,9,5,6,7]",
      "assert spiral_order([])==[]", "assert spiral_order([[1],[2],[3]])==[1,2,3]"]),
    ("min_window", "min_window(s: str, t: str) -> str: return the shortest substring of s containing every "
     "character of t with multiplicity, or '' if none. If several are shortest, return the leftmost.",
     "def min_window(s,t):\n    from collections import Counter\n    if not t: return ''\n    need=Counter(t); miss=len(t); best=''; l=0\n    for r,c in enumerate(s):\n        if need[c]>0: miss-=1\n        need[c]-=1\n        while miss==0:\n            if not best or r-l+1<len(best): best=s[l:r+1]\n            need[s[l]]+=1\n            if need[s[l]]>0: miss+=1\n            l+=1\n    return best",
     ["assert min_window('ADOBECODEBANC','ABC')=='BANC'", "assert min_window('a','aa')==''",
      "assert min_window('aa','aa')=='aa'", "assert min_window('abcabc','cba')=='abc'",
      "assert min_window('xyz','')==''"]),
    ("decode_string", "decode_string(s: str) -> str: expand k[text] patterns, which may nest, e.g. "
     "'3[a2[c]]' -> 'accaccacc'. k can have several digits.",
     "def decode_string(s):\n    st=[]; cur=''; k=0\n    for c in s:\n        if c.isdigit(): k=k*10+int(c)\n        elif c=='[': st.append((cur,k)); cur=''; k=0\n        elif c==']': p,n=st.pop(); cur=p+cur*n\n        else: cur+=c\n    return cur",
     ["assert decode_string('3[a2[c]]')=='accaccacc'", "assert decode_string('2[abc]3[cd]ef')=='abcabccdcdcdef'",
      "assert decode_string('10[a]')=='a'*10", "assert decode_string('abc')=='abc'",
      "assert decode_string('2[x0[y]]')=='xx'"]),
    ("next_permutation", "next_permutation(nums: list[int]) -> list[int]: return a new list holding the next "
     "lexicographic permutation; if nums is the last permutation return the sorted ascending list.",
     "def next_permutation(nums):\n    a=list(nums); i=len(a)-2\n    while i>=0 and a[i]>=a[i+1]: i-=1\n    if i<0: return sorted(a)\n    j=len(a)-1\n    while a[j]<=a[i]: j-=1\n    a[i],a[j]=a[j],a[i]\n    a[i+1:]=reversed(a[i+1:])\n    return a",
     ["assert next_permutation([1,2,3])==[1,3,2]", "assert next_permutation([3,2,1])==[1,2,3]",
      "assert next_permutation([1,1,5])==[1,5,1]", "assert next_permutation([1,3,2])==[2,1,3]",
      "assert next_permutation([2,3,1,3,3])==[2,3,3,1,3]", "assert next_permutation([])==[]"]),
    ("top_k_frequent", "top_k_frequent(words: list[str], k: int) -> list[str]: the k most frequent words, "
     "most frequent first; ties broken alphabetically.",
     "def top_k_frequent(words,k):\n    from collections import Counter\n    c=Counter(words)\n    return sorted(c,key=lambda w:(-c[w],w))[:k]",
     ["assert top_k_frequent(['i','love','leetcode','i','love','coding'],2)==['i','love']",
      "assert top_k_frequent(['the','day','is','sunny','the','the','the','sunny','is','is'],4)==['the','is','sunny','day']",
      "assert top_k_frequent(['b','a'],2)==['a','b']", "assert top_k_frequent([],3)==[]"]),
    ("simplify_path", "simplify_path(path: str) -> str: canonicalise an absolute Unix path ('.' and '..', "
     "repeated slashes, no trailing slash; '...' is a normal name).",
     "def simplify_path(path):\n    st=[]\n    for p in path.split('/'):\n        if p=='..':\n            if st: st.pop()\n        elif p and p!='.': st.append(p)\n    return '/'+'/'.join(st)",
     ["assert simplify_path('/home/')=='/home'", "assert simplify_path('/../')=='/'",
      "assert simplify_path('/home//foo/')=='/home/foo'", "assert simplify_path('/a/./b/../../c/')=='/c'",
      "assert simplify_path('/.../a/../b')=='/.../b'"]),
    ("calculate", "calculate(expr: str) -> int: evaluate an integer expression with + - * /, unary minus, "
     "parentheses and spaces. Division truncates toward zero. Do not use eval or exec.",
     "def calculate(expr):\n    s=expr.replace(' ','')\n    pos=0\n    def atom():\n        nonlocal pos\n        if s[pos]=='-': pos+=1; return -atom()\n        if s[pos]=='+': pos+=1; return atom()\n        if s[pos]=='(':\n            pos+=1; v=add(); pos+=1; return v\n        j=pos\n        while pos<len(s) and s[pos].isdigit(): pos+=1\n        return int(s[j:pos])\n    def mul():\n        nonlocal pos\n        v=atom()\n        while pos<len(s) and s[pos] in '*/':\n            o=s[pos]; pos+=1; w=atom()\n            v=v*w if o=='*' else int(v/w)\n        return v\n    def add():\n        nonlocal pos\n        v=mul()\n        while pos<len(s) and s[pos] in '+-':\n            o=s[pos]; pos+=1; w=mul()\n            v=v+w if o=='+' else v-w\n        return v\n    return add()",
     ["assert calculate('1 + 1')==2", "assert calculate(' 2-1 + 2 ')==3", "assert calculate('(1+(4+5+2)-3)+(6+8)')==23",
      "assert calculate('2*(5+5*2)/3+(6/2+8)')==21", "assert calculate('-(3+4)*2')==-14",
      "assert calculate('7/-2')==-3", "assert calculate('14-3/2')==13"]),
    ("count_islands", "count_islands(grid: list[list[str]]) -> int: count groups of '1' cells connected "
     "horizontally or vertically. Must not modify the input grid.",
     "def count_islands(grid):\n    seen=set(); n=0\n    for i in range(len(grid)):\n        for j in range(len(grid[0])):\n            if grid[i][j]=='1' and (i,j) not in seen:\n                n+=1; st=[(i,j)]; seen.add((i,j))\n                while st:\n                    a,b=st.pop()\n                    for x,y in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):\n                        if 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y]=='1' and (x,y) not in seen:\n                            seen.add((x,y)); st.append((x,y))\n    return n",
     ["g=[list('11000'),list('11000'),list('00100'),list('00011')]\nimport copy\nh=copy.deepcopy(g)\nassert count_islands(g)==3\nassert g==h",
      "assert count_islands([])==0", "assert count_islands([list('101'),list('010'),list('101')])==5",
      "assert count_islands([list('111'),list('101'),list('111')])==1"]),
    ("edit_distance", "edit_distance(a: str, b: str) -> int: Levenshtein distance (insert, delete, substitute).",
     "def edit_distance(a,b):\n    d=list(range(len(b)+1))\n    for i,x in enumerate(a,1):\n        p=d[:]; d[0]=i\n        for j,y in enumerate(b,1): d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))\n    return d[-1]",
     ["assert edit_distance('horse','ros')==3", "assert edit_distance('intention','execution')==5",
      "assert edit_distance('','abc')==3", "assert edit_distance('kitten','sitting')==3",
      "assert edit_distance('same','same')==0"]),
    ("largest_rectangle", "largest_rectangle(heights: list[int]) -> int: area of the largest rectangle in a "
     "histogram with unit-width bars. Must run in O(n) or O(n log n).",
     "def largest_rectangle(h):\n    st=[]; best=0\n    for i,x in enumerate(list(h)+[0]):\n        s=i\n        while st and st[-1][1]>=x:\n            s,y=st.pop(); best=max(best,y*(i-s))\n        st.append((s,x))\n    return best",
     ["assert largest_rectangle([2,1,5,6,2,3])==10", "assert largest_rectangle([2,4])==4",
      "assert largest_rectangle([])==0", "assert largest_rectangle([6,2,5,4,5,1,6])==12",
      "assert largest_rectangle(list(range(1,20001)))==100010000"]),
    ("trap_water", "trap_water(heights: list[int]) -> int: units of rain water trapped between the bars.",
     "def trap_water(h):\n    l,r=0,len(h)-1; a=b=t=0\n    while l<r:\n        if h[l]<h[r]: a=max(a,h[l]); t+=a-h[l]; l+=1\n        else: b=max(b,h[r]); t+=b-h[r]; r-=1\n    return t",
     ["assert trap_water([0,1,0,2,1,0,1,3,2,1,2,1])==6", "assert trap_water([4,2,0,3,2,5])==9",
      "assert trap_water([])==0", "assert trap_water([3,0,0,0,3])==9", "assert trap_water([1,2,3])==0"]),
    ("word_break", "word_break(s: str, words: list[str]) -> bool: can s be split into a sequence of one or "
     "more words from the list (reuse allowed)? The empty string returns True.",
     "def word_break(s,words):\n    w=set(words); ok=[True]+[False]*len(s)\n    for i in range(1,len(s)+1):\n        ok[i]=any(ok[j] and s[j:i] in w for j in range(i))\n    return ok[-1]",
     ["assert word_break('leetcode',['leet','code']) is True", "assert word_break('applepenapple',['apple','pen']) is True",
      "assert word_break('catsandog',['cats','dog','sand','and','cat']) is False", "assert word_break('',['a']) is True",
      "assert word_break('a'*40+'b',['a','aa','aaa','aaaa']) is False"]),
    ("course_order", "course_order(n: int, prereqs: list[list[int]]) -> list[int]: courses are 0..n-1 and "
     "[a, b] means b must be taken before a. Return any valid order of all n courses, or [] if impossible.",
     "def course_order(n,prereqs):\n    g=[[] for _ in range(n)]; deg=[0]*n\n    for a,b in prereqs: g[b].append(a); deg[a]+=1\n    q=[i for i in range(n) if deg[i]==0]; out=[]\n    while q:\n        x=q.pop(); out.append(x)\n        for y in g[x]:\n            deg[y]-=1\n            if deg[y]==0: q.append(y)\n    return out if len(out)==n else []",
     ["def ok(n,p,o):\n    pos={c:i for i,c in enumerate(o)}\n    return sorted(o)==list(range(n)) and all(pos[b]<pos[a] for a,b in p)\nassert ok(4,[[1,0],[2,0],[3,1],[3,2]],course_order(4,[[1,0],[2,0],[3,1],[3,2]]))",
      "assert course_order(2,[[0,1],[1,0]])==[]", "assert sorted(course_order(3,[]))==[0,1,2]",
      "assert course_order(3,[[0,1],[1,2],[2,0]])==[]", "assert course_order(1,[])==[0]"]),
    ("lis_length", "lis_length(nums: list[int]) -> int: length of the longest strictly increasing subsequence, "
     "in O(n log n).",
     "def lis_length(nums):\n    import bisect\n    t=[]\n    for x in nums:\n        i=bisect.bisect_left(t,x)\n        if i==len(t): t.append(x)\n        else: t[i]=x\n    return len(t)",
     ["assert lis_length([10,9,2,5,3,7,101,18])==4", "assert lis_length([0,1,0,3,2,3])==4",
      "assert lis_length([7,7,7,7])==1", "assert lis_length([])==0",
      "assert lis_length(list(range(50000)))==50000"]),
]


def run_tests(code, tests):
    """Run candidate code plus each test in a fresh, isolated interpreter. Returns True if all pass."""
    with tempfile.TemporaryDirectory() as d:
        for t in tests:
            try:
                p = subprocess.run([sys.executable, "-I", "-c", code + "\n" + t], cwd=d, capture_output=True, timeout=15)
            except subprocess.TimeoutExpired:
                return False
            if p.returncode != 0:
                return False
    return True


def selfcheck():
    bad = [n for n, _, ref, tests in TASKS if not run_tests(ref, tests)]
    print(f"{len(TASKS) - len(bad)}/{len(TASKS)} reference solutions pass their tests", "FAILED: " + str(bad) if bad else "")
    sys.exit(1 if bad else 0)


def fixed_text():
    import bisect, heapq, textwrap
    parts = [Path(m.__file__).read_text() for m in (heapq, textwrap, bisect)]
    parts.append((ROOT / "source" / "README.md").read_text())
    return "\n\n".join(parts)


def measure(build, stop_at, only=None, loss_only=False, sample=False, tasks=None, seed=11, label=None):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm import generate, load

    # --sample: the base model's recommended settings instead of temperature 0; saved separately
    name = (label or build.name) + (("-sampled" if seed == 11 else f"-sampled-s{seed}") if sample else "")
    gen = dict(temperature=1.0, top_p=0.95, top_k=20, seed=seed) if sample else dict(temperature=0.0)
    out_path = RES / f"quality-{name}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"build": name, "tasks": {}}
    model, processor = load(str(build))
    tok = processor.tokenizer

    if "loss" not in res and not sample:
        ids = tok.encode(fixed_text())[: 3 * 2048]
        nll, top = [], []
        for w in range(0, len(ids), 2048):
            x = mx.array(ids[w:w + 2048])[None]
            out = model.language_model(x)
            logits = (out.logits if hasattr(out, "logits") else out)[0, :-1].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = x[0, 1:]
            nll.append(np.array(-mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]))
            top.append(np.array(mx.argmax(logits, axis=-1)))
            mx.clear_cache()
        nll, top = np.concatenate(nll), np.concatenate(top)
        np.save(RES / f"top-{name}.npy", top)
        res["loss"] = {"tokens": int(nll.size), "mean_nll": round(float(nll.mean()), 4),
                       "perplexity": round(float(np.exp(nll.mean())), 4)}
        out_path.write_text(json.dumps(res, indent=2))
        print("loss", res["loss"], flush=True)

    if loss_only:
        return
    for tname, desc, _, tests in TASKS:
        if (only and tname != only) or (tasks and tname not in tasks):
            continue
        if tname in res["tasks"] and not only:
            continue
        if stop_at and datetime.datetime.now().strftime("%H:%M") >= stop_at:
            print("stopping at", stop_at, "- rerun to resume", flush=True)
            break
        msg = [{"role": "user", "content": "Write the Python function " + desc +
                " Use only the standard library. Give the final code in one ```python block."}]
        prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        r = generate(model, processor, prompt, max_tokens=4000, **gen)
        answer = r.text.split("</think>")[-1]
        blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.S)
        ok = bool(blocks) and run_tests(blocks[-1], tests)
        if only:  # diagnostic rerun: keep the full text, leave the saved results alone
            (RES / f"dump-{name}-{tname}.txt").write_text(r.text)
            print(f"{'PASS' if ok else 'FAIL'} {tname} {r.generation_tokens} tok (dump saved)", flush=True)
            return
        res["tasks"][tname] = {"pass": ok, "tokens": r.generation_tokens, "seconds": round(time.time() - t0, 1),
                               "tps": round(r.generation_tps, 1), "closed_thinking": "</think>" in r.text,
                               "end_think_tags": r.text.count("</think>"),
                               "answer": answer[-1500:]}
        out_path.write_text(json.dumps(res, indent=2))
        print(f"{'PASS' if ok else 'FAIL'} {tname} {r.generation_tokens} tok {time.time() - t0:.0f}s", flush=True)
    done = res["tasks"]
    print(f"{name}: {sum(t['pass'] for t in done.values())}/{len(done)} tasks passed ({len(TASKS)} total)")


def compare():
    import numpy as np
    ref = RES / "top-AREX-2-full.npy"   # full-precision original, when measured
    if not ref.exists():
        ref = RES / "top-AREX-2-8bit.npy"
    print("reference:", ref.name)
    for p in sorted(RES.glob("quality-*.json")):
        r = json.loads(p.read_text())
        t = r["tasks"]
        agree = ""
        tp = RES / f"top-{r['build']}.npy"
        if ref.exists() and tp.exists():
            agree = f"{(np.load(ref) == np.load(tp)).mean() * 100:.2f}%"
        toks = sum(x["tokens"] for x in t.values())
        print(f"{r['build']:14s} perplexity {r.get('loss', {}).get('perplexity')}  same top token as 8-bit {agree:8s} "
              f"tasks {sum(x['pass'] for x in t.values())}/{len(t)}  tokens {toks}  "
              f"failed: {[k for k, x in t.items() if not x['pass']]}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    elif "--compare" in sys.argv:
        compare()
    else:
        stop = sys.argv[sys.argv.index("--stop-at") + 1] if "--stop-at" in sys.argv else None
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        tasks = sys.argv[sys.argv.index("--tasks") + 1].split(",") if "--tasks" in sys.argv else None
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 11
        label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else None
        measure(Path(sys.argv[1]), stop, only, "--loss-only" in sys.argv, "--sample" in sys.argv, tasks, seed, label)
