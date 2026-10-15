"""Export a small review/submission bundle without data, weights, envs or keys."""
import zipfile
from pathlib import Path

def main():
    root = Path.cwd()
    candidates = []
    for directory in ("student","student_tests","scripts","tests","configs","report"):
        candidates += [p for p in (root/directory).rglob("*") if p.is_file()
                       and p.suffix in (".py",".md",".yaml",".json",".jsonl",".png")]
    candidates += [root/n for n in ("lab_day10.ipynb","REPORT.md","REPORT.generated.md","README.md",
                   "GUIDE.md","RUBRIC.md","KAGGLE.md","CHECKLIST.md","BAO_CAO_LOI_VA_LUU_Y.md",
                   "requirements-colab.txt","requirements-kaggle.txt","pytest.ini",".gitignore")]
    with zipfile.ZipFile(root/"submission_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in candidates:
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p,p.relative_to(root).as_posix())
        for p in (root/"work").glob("*_test.jsonl"):
            z.write(p,f"report/predictions/{p.name}")
    print("submission_bundle.zip created. Review REPORT.generated.md; no trained weights included.")

if __name__ == "__main__":
    main()
