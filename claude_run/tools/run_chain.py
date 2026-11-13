"""Build and execute the c-notebooks in order. Finished training runs are cached in results/, so a notebook
that already ran is re-rendered in seconds and an interrupted chain continues where it stopped.

    python tools/run_chain.py            # c6 .. c10
    python tools/run_chain.py 7 8        # only c7 and c8
"""
import shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
# "6" -> c6 (model notebooks), "d3" -> d3 (compression notebooks)
steps = [a if a.startswith("d") else f"c{a}" for a in sys.argv[1:]] or ["c6", "c7", "c8", "c9", "c10"]

# Keep everything: before a chain overwrites notebooks, artifacts and logs, copy them to history/.
# (Training runs are never overwritten either: lib.train moves replaced runs to results/_superseded/.)
snap = RUN / "history" / f"auto_{time.strftime('%Y%m%d-%H%M%S')}_before_{'-'.join(steps)}"
snap.mkdir(parents=True)
shutil.copytree(RUN / "artifacts", snap / "artifacts")
if (HERE / "_out").exists():
    shutil.copytree(HERE / "_out", snap / "logs")
for nb in [*RUN.glob("c*.ipynb"), *RUN.glob("d*.ipynb")]:
    shutil.copy2(nb, snap / nb.name)
print(f"snapshot: {snap}", flush=True)
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
