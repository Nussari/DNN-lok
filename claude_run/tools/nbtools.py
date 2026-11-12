import subprocess, sys, base64, json
from pathlib import Path
import nbformat as nbf

RUN = Path(__file__).resolve().parents[1]          # claude_run/
SP = Path(__file__).resolve().parent / "_out"      # extracted plot images
SP.mkdir(exist_ok=True)


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def write(self, name, execute=True):
        nb = nbf.v4.new_notebook(cells=self.cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = RUN / name
        nbf.write(nb, path)
        if execute:
            r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                "--ExecutePreprocessor.timeout=-1", str(path)], capture_output=True, text=True)
            if r.returncode:
                print(r.stderr[-3000:])
                raise SystemExit(f"{name}: execution failed")
        dump(name)


def dump(name, img_prefix=None):
    """Print text outputs of an executed notebook and save its images to the scratchpad."""
    nb = nbf.read(RUN / name, 4)
    stem = Path(name).stem
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                print(f"--[{i}]", o.text[:3000])
            elif o.output_type == "error":
                print(f"--[{i}] ERROR", o.ename, o.evalue)
            else:
                d = o.get("data", {})
                if "image/png" in d:
                    f = SP / f"{stem}_{i}.png"
                    f.write_bytes(base64.b64decode(d["image/png"]))
                    print(f"--[{i}] image -> {f.name}")
                elif "text/plain" in d:
                    print(f"--[{i}]", d["text/plain"][:3000])
