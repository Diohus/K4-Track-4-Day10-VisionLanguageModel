"""Conservative rubric audit: automatic evidence points, manual criteria kept pending."""
import math
from pathlib import Path
from scripts.check_submission import run_checks
from student.common import dump
from student.run import read


def main():
    evidence = {}
    for p in Path("report/evidence").glob("*.json"):
        evidence[p.stem] = read(p)
    version = read("data/splits/meta.json")["dataset_version"] if Path("data/splits/meta.json").exists() else None
    def valid(n):
        d = evidence.get(n,{})
        return (d.get("score_version")==1 and d.get("split")=="test"
                and d.get("missing_predictions")==0 and d.get("n_images")==311
                and d.get("dataset_version")==version and version is not None)
    def meta_ok(n):
        m = evidence.get(n,{}).get("meta",{})
        return all(isinstance(m.get(k),(int,float)) and not isinstance(m[k],bool)
                   and math.isfinite(m[k]) and m[k]>0
                   for k in ("trainable_params","peak_vram_gb","train_minutes"))
    points = {"e0_e3":sum(3 for n in ("e0","e1","e2","e3") if valid(n)),
              "meta_e1_e3":sum(2 for n in ("e1","e2","e3") if valid(n) and meta_ok(n))}
    e1, e0 = evidence.get("e1",{}), evidence.get("e0",{})
    points["e1_parse"] = 4 if valid("e1") and e1.get("parse_valid_rate",0)>=.95 else 0
    points["e1_gain"] = 6 if valid("e1") and valid("e0") and e1["micro"]["f1"]-e0["micro"]["f1"]>=.10 else 0
    e4 = [n for n in evidence if n.startswith("e4")]
    points["e4"] = 6 if e4 and all(valid(n) for n in e4) else 0
    checks = run_checks(Path.cwd())
    evidence_fails = [r for r in checks if r[0]=="FAIL" and
                      ("evidence" in r[1] or "meta" in r[1] or "dataset_version" in r[1])]
    points["consistent"] = 6 if not evidence_fails and points["e0_e3"]==12 and points["e4"]==6 else 0
    result = {"A_confirmed":sum(points.values()), "A_max":40, "A_breakdown":points,
              "B_C_D_bonus":"Require human review and actual run evidence; not awarded from source code alone.",
              "official_checks":checks}
    dump("report/self_grade.json",result)
    text = f"""# Tự chấm theo rubric
A tự động xác nhận được: **{sum(points.values())}/40** (chưa chạy thì không có điểm thực nghiệm).
Chi tiết: {points}.

B /20: đối chiếu REPORT.generated.md với evidence; xem sáu ảnh đúng/sai, ablation.json và log tài nguyên.
C /30: tác giả cần đọc, bổ sung diễn giải cá nhân cho 10 câu; mã sinh báo cáo không thay thế đánh giá nội dung.
D /10: cần ≥5 commit thật theo checkpoint, repo sạch và checker 0 FAIL. Không tính lịch sử giả.
Bonus /10: reward có test là phần chuẩn bị; chỉ nhận điểm chạy GRPO khi có log/evidence thật và phân tích CI/reward hacking.

Không thể xác nhận điểm tối đa trước khi có kết quả GPU và đánh giá thủ công.

## Checker gốc
"""
    text += "\n".join(f"- [{status}] {name}: {detail}" for status,name,detail in checks)
    Path("report/SELF_GRADE.md").write_text(text+"\n",encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
