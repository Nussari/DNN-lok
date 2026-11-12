"""Build and execute the c-notebooks in order. Finished training runs are cached in results/, so a notebook
that already ran is re-rendered in seconds and an interrupted chain continues where it stopped.

    python tools/run_chain.py            # c6 .. c10
    python tools/run_chain.py 7 8        # only c7 and c8
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
steps = [int(a) for a in sys.argv[1:]] or [6, 7, 8, 9, 10]
for n in steps:
    t0 = time.time()
    print(f"=== c{n} started {time.strftime('%H:%M')}", flush=True)
    log = HERE / "_out" / f"c{n}_log.txt"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, str(HERE / f"make_c{n}.py")], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    text = log.read_text(errors="replace")
    for line in text.splitlines():
        if any(k in line for k in ["chosen", "ERROR", "final learning", "best deep", "->"]):
            print("   ", line[:200])
    print(f"=== c{n} done in {(time.time() - t0) / 60:.0f} min (log: {log})", flush=True)
    if r.returncode or "ERROR" in text:
        sys.exit(f"c{n} failed, see {log}")
