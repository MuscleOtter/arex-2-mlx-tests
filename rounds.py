"""Feedback-rounds comparison: harder problems, up to three attempts, the model sees its failing test each round.

Usage:
  venv/bin/python rounds.py --selfcheck
  venv/bin/python rounds.py <model_dir> --label NAME [--seed 41]     # resumes from results/rounds-NAME-s41.json
  venv/bin/python rounds.py --compare

Same prompts, sampling (temperature 1.0, top-p 0.95, top-k 20), token limit and seeds for every model.
"""
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
RES = ROOT / "results"
MAX_ROUNDS, MAX_TOKENS = 3, 4000

# (name, description, reference solution, tests). Tests run in order; the first failure is shown to the model.
TASKS = [
    ("regex_match", "regex_match(s: str, p: str) -> bool: full-string match where '.' matches any one character "
     "and '*' means zero or more of the preceding element.",
     "def regex_match(s,p):\n    from functools import lru_cache\n    @lru_cache(None)\n    def f(i,j):\n        if j==len(p): return i==len(s)\n        m=i<len(s) and p[j] in (s[i],'.')\n        if j+1<len(p) and p[j+1]=='*': return f(i,j+2) or (m and f(i+1,j))\n        return m and f(i+1,j+1)\n    return f(0,0)",
     ["assert regex_match('aa','a') is False", "assert regex_match('aa','a*') is True", "assert regex_match('ab','.*') is True",
      "assert regex_match('aab','c*a*b') is True", "assert regex_match('mississippi','mis*is*p*.') is False",
      "assert regex_match('','a*b*') is True", "assert regex_match('ab','.*c') is False", "assert regex_match('aaa','a*a') is True",
      "assert regex_match('a'*25+'b','a*a*a*a*a*a*a*a*a*c') is False"]),
    ("text_justify", "text_justify(words: list[str], width: int) -> list[str]: pack words greedily into lines of "
     "exactly `width` characters. Extra spaces are spread between words as evenly as possible, with the left gaps "
     "getting the extras. The last line, and any line with one word, is left-justified and padded with spaces.",
     "def text_justify(words,width):\n    out=[];line=[];n=0\n    for w in words:\n        if n+len(w)+len(line)>width:\n            gaps=max(1,len(line)-1)\n            for i in range(width-n): line[i%gaps]+=' '\n            out.append(''.join(line)); line=[];n=0\n        line.append(w); n+=len(w)\n    out.append(' '.join(line).ljust(width))\n    return out",
     ["assert text_justify(['This','is','an','example','of','text','justification.'],16)==['This    is    an','example  of text','justification.  ']",
      "assert text_justify(['What','must','be','acknowledgment','shall','be'],16)==['What   must   be','acknowledgment  ','shall be        ']",
      "assert text_justify(['Science','is','what','we','understand','well','enough','to','explain','to','a','computer.','Art','is','everything','else','we','do'],20)==['Science  is  what we','understand      well','enough to explain to','a  computer.  Art is','everything  else  we','do                  ']",
      "assert text_justify(['a'],1)==['a']", "assert text_justify(['a','b','c','d','e'],3)==['a b','c d','e  ']"]),
    ("strong_password_steps", "strong_password_steps(s: str) -> int: a strong password has 6 to 20 characters, at "
     "least one lowercase letter, one uppercase letter and one digit, and no three identical characters in a row. "
     "Return the minimum number of single-character insertions, deletions or replacements needed to make s strong.",
     "def strong_password_steps(s):\n    miss=3-(any(c.islower() for c in s)+any(c.isupper() for c in s)+any(c.isdigit() for c in s))\n    n=len(s)\n    if n<6: return max(miss,6-n)\n    runs=[];i=0\n    while i<n:\n        j=i\n        while j<n and s[j]==s[i]: j+=1\n        if j-i>=3: runs.append(j-i)\n        i=j\n    if n<=20: return max(miss,sum(r//3 for r in runs))\n    d=n-20;left=d\n    for m in (0,1):\n        for k in range(len(runs)):\n            if left>=m+1 and runs[k]>=3 and runs[k]%3==m:\n                runs[k]-=m+1; left-=m+1\n    for k in range(len(runs)):\n        if runs[k]>=3 and left>0:\n            t=min(left,runs[k]-2); runs[k]-=t; left-=t\n    return d+max(miss,sum(r//3 for r in runs))",
     ["assert strong_password_steps('a')==5", "assert strong_password_steps('aA1')==3", "assert strong_password_steps('1337C0d3')==0",
      "assert strong_password_steps('aaa111')==2", "assert strong_password_steps('aaaaaa')==2", "assert strong_password_steps('')==6",
      "assert strong_password_steps('ABABABABABABABABABAB1')==2", "assert strong_password_steps('aaaaaaaaaaaaaaaaaaaaa')==7",
      "assert strong_password_steps('bbaaaaaaaaaaaaaaacccccc')==8", "assert strong_password_steps('1111111111')==3",
      "assert strong_password_steps('aaaabbbbccccddeeddeeddeedd')==8", "assert strong_password_steps('FFFFFFFFFFFFFFF11111111111111111111AAA')==23"]),
    ("is_number", "is_number(s: str) -> bool: True if s is a valid decimal number or integer, optionally followed by "
     "an exponent. A number is an optional sign, then digits with an optional dot (at least one digit somewhere), "
     "then optionally 'e' or 'E' with an optionally signed integer. No surrounding spaces or other characters.",
     "def is_number(s):\n    import re\n    return re.fullmatch(r'[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?',s) is not None",
     ["assert all(is_number(x) for x in ['2','0089','-0.1','+3.14','4.','-.9','2e10','-90E3','3e+7','+6e-1','53.5e93','-123.456e789'])",
      "assert not any(is_number(x) for x in ['abc','1a','1e','e3','99e2.5','--6','-+3','95a54e53','.','','+','4e+','.e1','1 ',' 1','Infinity','0x10','1_000'])"]),
    ("number_to_words", "number_to_words(n: int) -> str: English words for 0 <= n < 2**31, title case, single spaces, "
     "no 'and', no hyphens, e.g. 1234567 -> 'One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven'.",
     "def number_to_words(n):\n    if n==0: return 'Zero'\n    a='One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()\n    b='Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()\n    def f(x):\n        if x==0: return []\n        if x<20: return [a[x-1]]\n        if x<100: return [b[x//10-2]]+f(x%10)\n        return [a[x//100-1],'Hundred']+f(x%100)\n    out=[]\n    for v,w in ((10**9,'Billion'),(10**6,'Million'),(1000,'Thousand'),(1,'')):\n        if n>=v:\n            out+=f(n//v)+([w] if w else []); n%=v\n    return ' '.join(out)",
     ["assert number_to_words(0)=='Zero'", "assert number_to_words(123)=='One Hundred Twenty Three'",
      "assert number_to_words(12345)=='Twelve Thousand Three Hundred Forty Five'",
      "assert number_to_words(1234567)=='One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven'",
      "assert number_to_words(1000000)=='One Million'", "assert number_to_words(100)=='One Hundred'",
      "assert number_to_words(2147483647)=='Two Billion One Hundred Forty Seven Million Four Hundred Eighty Three Thousand Six Hundred Forty Seven'",
      "assert number_to_words(1000010)=='One Million Ten'", "assert number_to_words(50868)=='Fifty Thousand Eight Hundred Sixty Eight'",
      "assert number_to_words(19)=='Nineteen'"]),
    ("skyline", "skyline(buildings: list[list[int]]) -> list[list[int]]: each building is [left, right, height]. "
     "Return the skyline as key points [x, height] sorted by x, with no two consecutive points of the same height; "
     "the last point has height 0.",
     "def skyline(buildings):\n    import heapq\n    ev=sorted([(l,-h,r) for l,r,h in buildings]+[(r,0,0) for _,r,_ in buildings])\n    res=[[0,0]]; live=[(0,float('inf'))]\n    for x,nh,r in ev:\n        while live[0][1]<=x: heapq.heappop(live)\n        if nh: heapq.heappush(live,(nh,r))\n        if res[-1][1]!=-live[0][0]: res.append([x,-live[0][0]])\n    return res[1:]",
     ["assert skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]])==[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]",
      "assert skyline([[0,2,3],[2,5,3]])==[[0,3],[5,0]]", "assert skyline([])==[]", "assert skyline([[1,2,1],[1,2,2],[1,2,3]])==[[1,3],[2,0]]",
      "assert skyline([[0,5,7],[5,10,7],[5,10,12],[10,15,7],[15,20,7],[15,20,12],[20,25,7]])==[[0,7],[5,12],[10,7],[15,12],[20,7],[25,0]]",
      "assert skyline([[2,4,70],[3,8,30],[6,100,41],[7,15,70],[10,30,102],[15,25,76],[60,80,91],[70,90,72],[85,120,59]])==[[2,70],[4,30],[6,41],[7,70],[10,102],[30,41],[60,91],[80,72],[90,59],[120,0]]"]),
    ("shortest_palindrome", "shortest_palindrome(s: str) -> str: the shortest palindrome obtainable by adding "
     "characters only in front of s. Must handle strings of 200,000 characters in about a second.",
     "def shortest_palindrome(s):\n    t=s+'#'+s[::-1]; f=[0]*len(t)\n    for i in range(1,len(t)):\n        k=f[i-1]\n        while k and t[i]!=t[k]: k=f[k-1]\n        f[i]=k+(t[i]==t[k])\n    return s[f[-1]:][::-1]+s",
     ["assert shortest_palindrome('aacecaaa')=='aaacecaaa'", "assert shortest_palindrome('abcd')=='dcbabcd'",
      "assert shortest_palindrome('')==''", "assert shortest_palindrome('aba')=='aba'", "assert shortest_palindrome('abb')=='bbabb'",
      "import time\ns='a'*99999+'b'+'a'*100000\nt=time.time()\nr=shortest_palindrome(s)\nassert time.time()-t<5, 'too slow on 200,000 characters'\nassert r=='a'+s",
      "import time\ns='ab'*100000\nt=time.time()\nr=shortest_palindrome(s)\nassert time.time()-t<5, 'too slow on 200,000 characters'\nassert r=='b'+s"]),
    ("count_smaller", "count_smaller(nums: list[int]) -> list[int]: result[i] is how many elements to the right of "
     "nums[i] are strictly smaller than it. Must handle 100,000 elements in a few seconds.",
     "def count_smaller(nums):\n    import bisect\n    seen=[]; out=[]\n    for x in reversed(nums):\n        i=bisect.bisect_left(seen,x); out.append(i); seen.insert(i,x)\n    return out[::-1]",
     ["assert count_smaller([5,2,6,1])==[2,1,1,0]", "assert count_smaller([-1,-1])==[0,0]", "assert count_smaller([])==[]",
      "assert count_smaller([2,0,1])==[2,0,0]", "assert count_smaller([1,2,3,4])==[0,0,0,0]",
      "import random\nrandom.seed(5)\na=[random.randint(-50,50) for _ in range(300)]\nassert count_smaller(a)==[sum(y<x for y in a[i+1:]) for i,x in enumerate(a)]",
      "import time,random\nrandom.seed(6)\na=[random.randint(-10**4,10**4) for _ in range(100000)]\nt=time.time()\nr=count_smaller(a)\nassert time.time()-t<10, 'too slow on 100,000 elements'\nassert r[-1]==0 and len(r)==100000 and r[0]==sum(y<a[0] for y in a[1:])"]),
    ("sliding_median", "sliding_median(nums: list[int], k: int) -> list[float]: the median of each window of size k "
     "as the window slides left to right (mean of the two middle values when k is even). Must handle 100,000 "
     "elements with k = 5,000 in a few seconds.",
     "def sliding_median(nums,k):\n    import bisect\n    w=sorted(nums[:k]); out=[]\n    for i in range(k,len(nums)+1):\n        out.append((w[k//2]+w[(k-1)//2])/2)\n        if i==len(nums): break\n        w.pop(bisect.bisect_left(w,nums[i-k])); bisect.insort(w,nums[i])\n    return out",
     ["assert sliding_median([1,3,-1,-3,5,3,6,7],3)==[1,-1,-1,3,5,6]", "assert sliding_median([1,2,3,4,2,3,1,4,2],3)==[2,3,3,3,2,3,2]",
      "assert sliding_median([1,4,2,3],4)==[2.5]", "assert sliding_median([5],1)==[5]", "assert sliding_median([2147483647,2147483647],2)==[2147483647]",
      "import random,statistics\nrandom.seed(8)\na=[random.randint(-99,99) for _ in range(200)]\nfor k in (1,2,7,10):\n    assert sliding_median(a,k)==[statistics.median(a[i:i+k]) for i in range(len(a)-k+1)], k",
      "import time,random\nrandom.seed(9)\na=[random.randint(-10**6,10**6) for _ in range(100000)]\nt=time.time()\nr=sliding_median(a,5000)\nassert time.time()-t<10, 'too slow on 100,000 elements'\nassert len(r)==95001"]),
    ("burst_balloons", "burst_balloons(nums: list[int]) -> int: bursting balloon i earns nums[left]*nums[i]*nums[right] "
     "coins, where left and right are its current neighbours (treat missing neighbours as 1). Return the maximum "
     "total coins from bursting all balloons. Up to 150 balloons.",
     "def burst_balloons(nums):\n    a=[1]+[x for x in nums if x>0]+[1]; n=len(a); d=[[0]*n for _ in range(n)]\n    for L in range(2,n):\n        for i in range(n-L):\n            j=i+L\n            d[i][j]=max(d[i][k]+d[k][j]+a[i]*a[k]*a[j] for k in range(i+1,j))\n    return d[0][n-1]",
     ["assert burst_balloons([3,1,5,8])==167", "assert burst_balloons([1,5])==10", "assert burst_balloons([])==0",
      "assert burst_balloons([7])==7", "assert burst_balloons([9,76,64,21,97,60])==1086136",
      "assert burst_balloons([0,3,0,5])==20",
      "import time\nt=time.time()\nr=burst_balloons([(i*37)%100 for i in range(150)])\nassert time.time()-t<20, 'too slow on 150 balloons'\nassert r>0"]),
    ("trap_rain_2d", "trap_rain_2d(heights: list[list[int]]) -> int: volume of water trapped on a 2D elevation map "
     "after rain (water can flow off the edges; cells connect up, down, left and right).",
     "def trap_rain_2d(h):\n    import heapq\n    if not h or not h[0]: return 0\n    m,n=len(h),len(h[0]); seen=[[False]*n for _ in range(m)]; pq=[]\n    for i in range(m):\n        for j in range(n):\n            if i in (0,m-1) or j in (0,n-1): heapq.heappush(pq,(h[i][j],i,j)); seen[i][j]=True\n    t=0\n    while pq:\n        x,i,j=heapq.heappop(pq)\n        for a,b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):\n            if 0<=a<m and 0<=b<n and not seen[a][b]:\n                seen[a][b]=True; t+=max(0,x-h[a][b]); heapq.heappush(pq,(max(x,h[a][b]),a,b))\n    return t",
     ["assert trap_rain_2d([[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]])==4",
      "assert trap_rain_2d([[3,3,3,3,3],[3,2,2,2,3],[3,2,1,2,3],[3,2,2,2,3],[3,3,3,3,3]])==10", "assert trap_rain_2d([])==0",
      "assert trap_rain_2d([[5,5,5],[5,1,5]])==0", "assert trap_rain_2d([[12,13,1,12],[13,4,13,12],[13,8,10,12],[12,13,12,12],[13,13,13,13]])==14",
      "assert trap_rain_2d([[9,9,9,9],[9,1,1,9],[9,1,9,9],[9,9,2,9],[9,9,9,9]])==31"]),
    ("max_points_on_line", "max_points_on_line(points: list[list[int]]) -> int: the largest number of the given "
     "points (integer coordinates, duplicates possible and counted separately) that lie on one straight line.",
     "def max_points_on_line(points):\n    from math import gcd\n    from collections import Counter\n    best=0\n    for i,(x,y) in enumerate(points):\n        c=Counter(); same=1\n        for a,b in points[i+1:]:\n            dx,dy=a-x,b-y\n            if dx==0 and dy==0: same+=1; continue\n            g=gcd(dx,dy); dx//=g; dy//=g\n            if dx<0 or (dx==0 and dy<0): dx,dy=-dx,-dy\n            c[(dx,dy)]+=1\n        best=max(best,same+(max(c.values()) if c else 0))\n    return best",
     ["assert max_points_on_line([[1,1],[2,2],[3,3]])==3", "assert max_points_on_line([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]])==4",
      "assert max_points_on_line([])==0", "assert max_points_on_line([[0,0]])==1", "assert max_points_on_line([[1,1],[1,1],[2,3]])==3",
      "assert max_points_on_line([[0,0],[94911151,94911150],[94911152,94911151]])==2", "assert max_points_on_line([[2,3],[2,9],[2,-4],[5,5]])==3",
      "assert max_points_on_line([[0,0],[1,1],[0,0],[2,2],[5,1],[1,1]])==5"]),
]


def run_tests(code, tests):
    """Returns (passed, feedback). Feedback is the first failing test and the error it raised."""
    with tempfile.TemporaryDirectory() as d:
        for t in tests:
            try:
                p = subprocess.run([sys.executable, "-I", "-c", code + "\n" + t], cwd=d, capture_output=True, text=True, timeout=40)
            except subprocess.TimeoutExpired:
                return False, f"Test:\n{t}\n\nResult: timed out after 40 seconds."
            if p.returncode != 0:
                err = "\n".join(p.stderr.strip().splitlines()[-6:])[-900:]
                return False, f"Test:\n{t[:700]}\n\nResult:\n{err}"
    return True, ""


def repetition(text):
    """How circular the output is: end-of-thinking tags, and the share of longer lines that are exact repeats."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 25]
    dup = len(lines) - len(set(lines))
    think = text.split("</think>")[0] if "</think>" in text else text
    return {"end_think_tags": text.count("</think>"), "repeated_line_share": round(dup / max(len(lines), 1), 3),
            "thinking_chars": len(think), "answer_chars": len(text) - len(think)}


def selfcheck():
    bad = []
    for n, _, ref, tests in TASKS:
        ok, fb = run_tests(ref, tests)
        if not ok:
            bad.append(n)
            print("REFERENCE FAILS", n, "\n", fb)
    print(f"{len(TASKS) - len(bad)}/{len(TASKS)} reference solutions pass their tests")
    sys.exit(1 if bad else 0)


def measure(model_dir, label, seed, max_rounds=MAX_ROUNDS, max_tokens=MAX_TOKENS, dump=False):
    from mlx_vlm import generate, load
    out_path = RES / f"rounds-{label}-s{seed}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"model": label, "seed": seed, "tasks": {}}
    model, processor = load(str(model_dir))
    tok = processor.tokenizer
    for ti, (name, desc, _, tests) in enumerate(TASKS):
        if name in res["tasks"]:
            continue
        msgs = [{"role": "user", "content": "Write the Python function " + desc +
                 " Use only the standard library. Give the complete code in one ```python block."}]
        rounds, solved = [], 0
        for rnd in range(1, max_rounds + 1):
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            t0 = time.time()
            r = generate(model, processor, prompt, max_tokens=max_tokens, temperature=1.0, top_p=0.95, top_k=20,
                         seed=seed * 1000 + ti * 10 + rnd)
            answer = r.text.split("</think>")[-1]
            blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.S)
            ok, fb = run_tests(blocks[-1], tests) if blocks else (False, "No ```python code block was found in the reply.")
            rounds.append({"round": rnd, "pass": ok, "tokens": r.generation_tokens, "seconds": round(time.time() - t0, 1),
                           "feedback": fb[:400], "hit_limit": r.generation_tokens >= max_tokens,
                           **repetition(r.text)})
            if dump:
                (RES / f"dump-rounds-{label}-{name}-r{rnd}.txt").write_text(r.text)
            print(f"{label} {name} round {rnd}: {'PASS' if ok else 'FAIL'} {r.generation_tokens} tok {time.time() - t0:.0f}s", flush=True)
            if ok:
                solved = rnd
                break
            msgs += [{"role": "assistant", "content": answer.strip()[-6000:] or "(no answer)"},
                     {"role": "user", "content": "That code failed this test.\n\n" + fb +
                      "\n\nFix it. Give the complete corrected code in one ```python block."}]
        res["tasks"][name] = {"solved_in_round": solved, "rounds": rounds}
        out_path.write_text(json.dumps(res, indent=2))
    t = res["tasks"]
    print(f"{label} seed {seed}: solved by round 1/2/3 = " +
          "/".join(str(sum(1 for v in t.values() if 0 < v['solved_in_round'] <= k)) for k in (1, 2, 3)) + f" of {len(t)}")


def compare():
    for p in sorted(RES.glob("rounds-*.json")):
        r = json.loads(p.read_text())
        t = r["tasks"]
        by = [sum(1 for v in t.values() if 0 < v["solved_in_round"] <= k) for k in (1, 2, 3)]
        toks = sum(x["tokens"] for v in t.values() for x in v["rounds"])
        print(f"{r['model']:22s} seed {r['seed']}  solved by round 1/2/3: {by[0]}/{by[1]}/{by[2]} of {len(t)}  tokens {toks}  "
              f"unsolved: {[k for k, v in t.items() if not v['solved_in_round']]}")


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    if "--selfcheck" in sys.argv:
        selfcheck()
    elif "--compare" in sys.argv:
        compare()
    else:
        label = sys.argv[sys.argv.index("--label") + 1]
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 41
        mr = int(sys.argv[sys.argv.index("--max-rounds") + 1]) if "--max-rounds" in sys.argv else MAX_ROUNDS
        mt = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else MAX_TOKENS
        measure(Path(sys.argv[1]), label, seed, mr, mt, "--dump" in sys.argv)
