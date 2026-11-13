"""Build and execute the pipeline notebooks in order. Finished training runs are cached in results/, so a notebook
that already ran is re-rendered in seconds and an interrupted chain continues where it stopped.

    python tools/run_chain.py 1 2 3 4 5 6 7 8 9 d1 d2 d3 d4 d5 d6 10 test

c10 applies the tumour decision rule to the compressed models picked in d6, so it runs after d6.
    python tools/run_chain.py 7 8        # only c7 and c8

Logs go to tools/_out/<step>_log.txt.
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# "6" -> c6 (classifier notebooks), "d3" -> d3 (compression notebooks), "test" -> the test evaluation
steps = [a if a.startswith("d") or a == "test" else f"c{a}" for a in sys.argv[1:]]
if not steps:
    sys.exit(__doc__)

for n in steps:
    t0 = time.time()
    print(f"=== {n} started {time.strftime('%H:%M')}", flush=True)
    log = HERE / "_out" / f"{n}_log.txt"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, str(HERE / f"make_{n}.py")], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    text = log.read_text(errors="replace")
    for line in text.splitlines():
        if any(k in line for k in ["chosen", "ERROR", "final learning", "best deep", "->"]):
            print("   ", line[:200])
    print(f"=== {n} done in {(time.time() - t0) / 60:.0f} min (log: {log})", flush=True)
    if r.returncode or "ERROR" in text:
        sys.exit(f"{n} failed, see {log}")
