"""Build measured report and annotated figures after GPU runs; never invent evidence."""
import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _prf, evaluate, parse_output
from student.common import dump, objects_to_answer
from student.run import read

STUDENT = "Nguyễn Khánh Đô"


def predictions(name, split="test"):
    rows = read_jsonl(f"work/{name}_{split}.jsonl")
    result = {r["image_id"]: r["raw"] for r in rows}
    assert len(result) == len(rows), "Duplicate prediction IDs"
    return result


def image_counts(entry, raw):
    per, _ = _match(_gt_boxes(entry), parse_output(raw) or [])
    return tuple(sum(v[i] for v in per.values()) for i in range(3))


def paired_ci(entries, a, b, unit="session", repeats=2000):
    """Paired difference F1(B)-F1(A), resample same units for both setups."""
    groups = {}
    for row in entries:
        key = row["session"] if unit == "session" else row["image_id"]
        group = groups.setdefault(key, [[0, 0, 0], [0, 0, 0]])
        for j, pred in enumerate((a, b)):
            counts = image_counts(row, pred[row["image_id"]])
            for i in range(3):
                group[j][i] += counts[i]
    groups = list(groups.values())
    rng = random.Random(42)
    samples = []
    for _ in range(repeats):
        totals = [[0, 0, 0], [0, 0, 0]]
        for _ in groups:
            chosen = rng.choice(groups)
            for j in range(2):
                for i in range(3):
                    totals[j][i] += chosen[j][i]
        samples.append(_prf(*totals[1])["f1"]-_prf(*totals[0])["f1"])
    samples.sort()
    return {"unit": unit, "groups": len(groups), "replicates": repeats,
            "delta_ci95": [samples[int(.025*repeats)], samples[int(.975*repeats)-1]]}


def draw_boxes(entry, raw, output, caption):
    with Image.open(Path("data") / entry["image_path"]) as im:
        im = im.convert("RGB")
    canvas = Image.new("RGB", (im.width, im.height+65), "white")
    canvas.paste(im)
    draw = ImageDraw.Draw(canvas)
    for boxes, color, prefix in ((_gt_boxes(entry), "lime", "GT"), (parse_output(raw) or [], "red", "P")):
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            xy = [x1/1000*im.width, y1/1000*im.height, x2/1000*im.width, y2/1000*im.height]
            draw.rectangle(xy, outline=color, width=2)
            draw.text((max(0, xy[0]), max(0, xy[1]-12)), prefix+":"+box["label"], fill=color)
    draw.text((5, im.height+4), caption[:100], fill="black")
    draw.text((5, im.height+22), "Green = ground truth; red = prediction; coordinates 0-1000", fill="black")
    draw.text((5, im.height+40), entry["image_id"][:95], fill="black")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def preview():
    rows = read_jsonl("data/splits/train.jsonl")[:6]
    audit = []
    for i, e in enumerate(rows):
        answer = objects_to_answer(e["objects"])
        gt, converted = _gt_boxes(e), parse_output(answer)
        # Compare boxes by label and sorted coordinates; half-unit rounding <= .32 px.
        original = sorted(gt, key=lambda b: (b["bbox"][1], b["bbox"][0], b["label"], b["bbox"][2:]))
        errors = []
        # Match each rounded box to nearest original of same class, without reusing GT.
        remaining = list(original)
        for b in converted:
            candidates = [g for g in remaining if g["label"] == b["label"]]
            g = min(candidates, key=lambda g: sum(abs(x-y) for x,y in zip(g["bbox"],b["bbox"])))
            remaining.remove(g)
            errors += [abs(x-y)/1000*(e["width"] if j%2 == 0 else e["height"])
                       for j,(x,y) in enumerate(zip(g["bbox"], b["bbox"]))]
        draw_boxes(e, answer, f"report/figures/ground_truth_{i+1}.png", "YOLO round-trip: green original, red rounded")
        audit.append({"image_id": e["image_id"], "max_pixel_error": max(errors, default=0)})
    dump("report/diagnostics/round_trip.json", audit)


def error_summary(entries, pred):
    counts = Counter()
    examples = {}
    for e in entries:
        raw = pred[e["image_id"]]
        parsed = parse_output(raw)
        tp, fp, fn = image_counts(e, raw)
        tags = []
        if parsed is None:
            tags.append("invalid_json")
        else:
            if fp:
                tags.append("unmatched_prediction")
            if fn:
                tags.append("missed_ground_truth")
            if any(p["label"] not in CLASSES for p in parsed):
                tags.append("unknown_class")
            if any(any(v < 0 or v > 1000 for v in p["bbox"]) for p in parsed):
                tags.append("out_of_scale")
            if not parsed and e["objects"]:
                tags.append("empty_on_positive_image")
        for tag in tags:
            counts[tag] += 1
            examples.setdefault(tag, {"image_id": e["image_id"], "raw": raw, "tp": tp, "fp": fp, "fn": fn})
    return {"counts_images": dict(counts), "examples": examples}


def hacking(entries, pred):
    parsed = [parse_output(pred[e["image_id"]]) for e in entries]
    boxes = [p for row in parsed if row is not None for p in row]
    return {"empty_rate": sum(p == [] for p in parsed)/len(parsed),
            "invalid_rate": sum(p is None for p in parsed)/len(parsed),
            "giant_box_fraction": sum((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) >= 800000
                                      for p in boxes)/max(1, len(boxes)),
            "predicted_boxes_per_image": len(boxes)/len(entries)}


def generate(student=STUDENT):
    names = ["e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0"]
    if Path("report/evidence/e5.json").exists():
        names.append("e5")
    evidence = {n: read(f"report/evidence/{n}.json") for n in names}
    meta = read("data/splits/meta.json")
    entries = read_jsonl("data/splits/test.jsonl")
    pred = {n: predictions(n) for n in names}
    ids = {e["image_id"] for e in entries}
    for n, d in evidence.items():
        assert d["dataset_version"] == meta["dataset_version"]
        assert d["split"] == "test" and d["missing_predictions"] == 0 and set(pred[n]) == ids
        reproduced = evaluate(entries, pred[n])
        for key in ("micro", "f1_ci95", "per_class", "parse_valid_rate"):
            assert d[key] == reproduced[key], f"Evidence/predictions mismatch: {n}/{key}"
    best = read("report/selection.json")["best"]
    runs = {n: read(f"report/runs/{n}.json") for n in ("e1", "e2", "e3")}
    for n, run in runs.items():
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            assert run[key] == evidence[n]["meta"][key]
    errors = {n: error_summary(entries, pred[n]) for n in names}
    dump("report/diagnostics/errors.json", errors)
    # Explicit coordinate counterfactual, outside official evidence directory.
    wrong_scale = evaluate(entries, pred["e1"], coord="pixel")
    dump("report/diagnostics/e1_wrong_pixel_scale.json", wrong_scale)
    comparisons = {}
    pairs = list(itertools.combinations(["e1", "e2", "e3"], 2))+[("e4_data_size_0.1", "e4_data_size_1.0")]
    if "e5" in names:
        pairs.append((best, "e5"))
    for a, b in pairs:
        comparisons[f"{b}-{a}"] = {
            "delta_f1": evidence[b]["micro"]["f1"]-evidence[a]["micro"]["f1"],
            "image": paired_ci(entries, pred[a], pred[b], "image"),
            "session": paired_ci(entries, pred[a], pred[b], "session")}
    dump("report/diagnostics/paired_ci.json", comparisons)

    scored = [(e, image_counts(e, pred[best][e["image_id"]])) for e in entries]
    successes = [(e,c) for e,c in scored if c[0] > 0 and c[1] == c[2] == 0]
    failures = [(e,c) for e,c in scored if c[1]+c[2] > 0]
    failures.sort(key=lambda ec: (-(ec[1][1]+ec[1][2]), ec[0]["image_id"]))
    selected = successes[:3]+failures[:3]
    used = {e["image_id"] for e,_ in selected}
    selected += [(e,c) for e,c in scored if e["image_id"] not in used][:6-len(selected)]
    figure_lines, figure_meta = [], []
    for i,(e,c) in enumerate(selected):
        tp, fp, fn = c
        path = f"report/figures/prediction_{i+1}.png"
        caption = f"{best.upper()} | TP={tp}, FP={fp}, FN={fn}"
        draw_boxes(e, pred[best][e["image_id"]], path, caption)
        figure_lines.append(f"![{caption}]({path})\n\n{e['image_id']}: {caption}.")
        figure_meta.append({"path": path, "image_id": e["image_id"], "tp": tp, "fp": fp, "fn": fn})
    dump("report/diagnostics/figures.json", figure_meta)

    table = ["| Mã | Setup | Tham số train | VRAM GiB | Phút | F1 | CI95 | FP/ảnh | pothole | crack | manhole |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    descriptions = {"e0":"zero-shot", "e1":"LoRA LLM", "e2":"+ full projector", "e3":"+ LoRA ViT",
                    "e4_data_size_0.1":f"{best}, 10% train", "e4_data_size_1.0":f"{best}, 100% train (reuse)",
                    "e5":f"GRPO từ {best}"}
    for n,d in evidence.items():
        m = d["meta"]
        vals = [n, descriptions[n], str(m.get("trainable_params",0)),
                f"{m.get('peak_vram_gb',0):.6f}", f"{m.get('train_minutes',0):.6f}",
                f"{d['micro']['f1']:.6f}", str(d["f1_ci95"]), f"{d['fp_per_image']:.6f}"]
        vals += [f"{d['per_class'][c]['f1']:.6f}" for c in CLASSES]
        table.append("| "+" | ".join(vals)+" |")
    mask = read("report/diagnostics/mask_audit.json")
    split_stats = read("report/diagnostics/data_audit.json")["splits"]
    quality_path = Path("report/diagnostics/zero_area_annotations.json")
    zero_area_count = read(quality_path)["count"] if quality_path.exists() else "chưa audit"
    worst = min(CLASSES, key=lambda c:evidence[best]["per_class"][c]["f1"])
    areas = {c: sorted(o["yolo"][2]*o["yolo"][3] for e in entries for o in e["objects"] if o["label"]==c) for c in CLASSES}
    area_summary = {c: __import__("statistics").median(v) for c,v in areas.items()}
    ci_text = "\n".join(f"- {n}: F1={d['micro']['f1']:.6f}, CI95={d['f1_ci95']}, độ rộng={d['f1_ci95'][1]-d['f1_ci95'][0]:.6f}."
                        for n,d in evidence.items())
    pair_text = "\n".join(
        f"- {n}: ΔF1={d['delta_f1']:.6f}; paired CI95 theo ảnh={d['image']['delta_ci95']}; "
        f"theo session ({d['session']['groups']} nhóm)={d['session']['delta_ci95']}."
        for n,d in comparisons.items())
    e0err = errors["e0"]
    top_errors = sorted(e0err["counts_images"].items(), key=lambda x:-x[1])[:2]
    ex = next(iter(e0err["examples"].values()), {"image_id":entries[0]["image_id"], "raw":pred["e0"][entries[0]["image_id"]]})
    delta = evidence["e1"]["micro"]["f1"]-evidence["e0"]["micro"]["f1"]
    text = f"""# Báo cáo Lab Day 10
Họ tên: {student}

dataset_version: {meta['dataset_version']}

Bản nháp phân tích được sinh từ kết quả thật; tác giả cần đọc ảnh, kiểm tra các giả thuyết và bổ sung nhận xét cá nhân trước khi nộp. Không coi việc checker qua là đủ điểm báo cáo.

## Bảng tổng hợp
{chr(10).join(table)}

Số trong bảng làm tròn 6 chữ số; bằng chứng JSON giữ toàn bộ độ chính xác. E0 không huấn luyện nên ba cột tài nguyên huấn luyện bằng 0; thời gian inference ở report/diagnostics. VRAM dùng max_memory_allocated / 2**30 (GiB), không phải tổng reserved hoặc số hiển thị nvidia-smi.

### Câu 1 — Kiến trúc và các khối cập nhật
Số phần tử trọng số gốc đo trước LoRA (khôi phục shape logic của Params4bit): {json.dumps(runs['e1']['architecture'])}.
vision_backbone loại trừ projector để tránh đếm hai lần; language_and_head gồm embedding và lm_head.
E1 chỉ LoRA LLM; E2 thêm full visual.merger và deepstack_merger_list; E3 thêm LoRA attention/MLP của ViT, gồm cả attn.proj.
Số module khớp: { {n:len(r['matched_modules']) for n,r in runs.items()} }.
Chênh lệch tham số train E2−E1: {runs['e2']['trainable_params']-runs['e1']['trainable_params']}.

### Câu 2 — E0
parse_valid_rate={evidence['e0']['parse_valid_rate']:.6f}. Hai nhóm lỗi tự động thường gặp nhất: {top_errors}.
Các nhóm có thể trùng trên cùng ảnh; unmatched_prediction không tự xác định được nguyên nhân là sai lớp hay lệch box.
Ví dụ ảnh {ex['image_id']}:
~~~text
{ex['raw']}
~~~
Cần xem ảnh và đối chiếu nhãn để xác định nguyên nhân cụ thể; không suy ra hallucination chỉ từ FP.

### Câu 3 — Thang 0–1000
Với cùng output E1, chấm đúng norm1000: F1={evidence['e1']['micro']['f1']:.6f}; cố tình chấm nhầm pixel: F1={wrong_scale['micro']['f1']:.6f}.
Pixel được đổi bằng x/width*1000 và y/height*1000. Nhầm đơn vị làm box dịch và phóng khác nhau theo hai trục, nên mất ghép IoU≥0.5.
Đây là kiểm tra cách chấm trên cùng prediction, không phải thí nghiệm huấn luyện pixel của E4.
Kiểm tra YOLO→JSON→pixel: xem report/diagnostics/round_trip.json và sáu ảnh ground_truth.
Thang chuẩn hóa giúp biểu diễn tương đối độc lập kích thước ảnh; lượng tử 1 đơn vị tương ứng 0.64 px ngang, 0.36 px dọc với ảnh này.

### Câu 4 — Mask loss
Audit thật: {mask['image_tokens']} token ảnh; {mask['total_tokens']} token tổng; {mask['supervised_tokens']} token được tính loss.
Prefix prompt đã được kiểm tra bằng tensor equality. Prompt, image tokens và padding mang nhãn −100; chỉ JSON assistant và im_end được giám sát.
Token giám sát giải mã:
~~~text
{mask['supervised_text']}
~~~
Nếu tính loss trên prompt/image placeholder, mô hình học dự đoán các token cấu trúc lặp lại và giảm trọng số tương đối của nhiệm vụ JSON.
Đây không phải loss tái tạo pixel ảnh. Mask nhãn cũng không cắt đường gradient qua đặc trưng ảnh: E3 vẫn học ViT từ loss câu trả lời.

### Câu 5 — Lượng tử hóa và projector
NF4 chỉ áp dụng cho LLM; visual và lm_head nằm trong skip_modules, có assertion không chứa Params4bit.
Projector full train cần trọng số floating-point có gradient; weight packed 4-bit không thể cập nhật trực tiếp như tham số dense thông thường.
Toàn bộ tham số train được giữ float32 (kể cả LoRA); forward dùng autocast fp16 và GradScaler. Điều này tránh lỗi unscale gradient fp16 và cải thiện độ ổn định cập nhật; float32 không bảo đảm hết mọi overflow.
Model lưu adapter và projector.pt riêng; inference nạp lại và kiểm tra tên/shape projector.

### Câu 6 — Split theo session
Số đo thực tế: {json.dumps(split_stats, ensure_ascii=False)}.
audit_data kiểm tra không giao session, không trùng image_id và đúng số box từng lớp.
Số nhãn nguồn có width hoặc height bằng 0 trên toàn bộ splits: {zero_area_count}; chi tiết ở report/diagnostics/zero_area_annotations.json.
Không xóa nhãn hoặc sửa splits. Khi serialize đáp án SFT, mở rộng trục suy biến thành 1 đơn vị trên thang 0–1000.
Scorer vẫn dùng GT gốc: GT diện tích 0 không thể ghép IoU≥0.5, tạo FN không thể khắc phục bằng dự đoán; đây là giới hạn chất lượng nhãn.
Chia frame ngẫu nhiên có thể đưa các ảnh gần trùng vào train/test, khiến F1 đánh giá khả năng nhớ bối cảnh thay vì tổng quát hóa trên phiên mới.
Chưa chạy random split nên không gán một mức tăng F1 giả định.

### Câu 7 — E1/E2/E3
Setup chọn trước khi xem test: {best}, theo micro F1 trên validation; xem report/selection.json.
Bảng trên ghi đầy đủ F1, chi phí VRAM/thời gian và F1 từng lớp. Mở thêm tham số chỉ đáng dùng nếu lợi ích phù hợp ngân sách và đủ chắc chắn.
{pair_text}
ΔF1 dương nghiêng về setup đứng trước dấu trừ. CI chứa 0 chưa đủ bằng chứng cải thiện. CI không chứa 0 chỉ hỗ trợ kết luận dưới cách lấy mẫu đã dùng; chưa đo biến thiên giữa các seed huấn luyện.

### Câu 8 — Độ bất định trên tập test
{ci_text}
CI chính thức bootstrap theo ảnh, nhưng các frame trong cùng session phụ thuộc nhau. Vì vậy tôi bổ sung paired bootstrap theo session, lấy cùng mẫu cho cả hai mô hình.
Chỉ có ít session độc lập nên CI theo nhóm cũng có thể không ổn định. Không dùng việc hai CI riêng lẻ giao nhau làm phép kiểm định chênh lệch; dùng CI của ΔF1 ở Câu 7.
Các kết quả là một seed, không chứng minh thứ hạng đúng cho mọi lần train hay mọi tuyến đường.

### Câu 9 — Lỗi theo lớp
Trong setup {best} chọn bằng val, lớp có F1 thấp nhất trên test là {worst}: {evidence[best]['per_class'][worst]['f1']:.6f}.
Trung vị diện tích box / diện tích ảnh trên test: {area_summary}. Số mẫu train từng lớp ở Câu 6.
Box nhỏ khiến dịch vài pixel làm IoU giảm mạnh; crack dài/mảnh có thể khó xác định biên. Đây là giả thuyết cần tác giả kiểm chứng trên ảnh, không phải nguyên nhân đã được chứng minh.
Các ảnh dưới đây được chọn từ tối đa ba ảnh khớp hoàn toàn và ba ảnh lỗi nhiều nhất; nếu thiếu nhóm nào thì bổ sung ảnh khác, không bịa nhãn đúng/sai.

{chr(10).join(figure_lines)}

### Câu 10 — E4
Chỉ đổi train_fraction 0.1 so với 1.0 trên setup {best}; 10% chọn xác định theo seed, giữ nguyên LR, rank, alpha, batch, số bước, prompt và decoding.
100% tái sử dụng chính run gốc; report/ablation.json chứa cả hai cấu hình và kiểm tra chỉ một trường khác.
Giữ 300 optimizer steps nên 10% dữ liệu bị lặp nhiều hơn; kết luận là ảnh hưởng độ đa dạng dữ liệu dưới ngân sách bước cố định, không phải số epoch cố định.
Chênh lệch và paired CI ở Câu 7; không kết luận chắc chắn nếu CI chứa 0.

## Tự kiểm tra các ngưỡng
E1 valid ≥0.95: {evidence['e1']['parse_valid_rate'] >= .95}.
F1(E1)−F1(E0)={delta:.6f}; ≥0.10: {delta >= .10}.
Nếu không đạt, chẩn đoán và điều chỉnh bằng validation; không sửa evidence hoặc chọn checkpoint bằng test.

## Nhìn lại
Tác giả cần bổ sung trải nghiệm trực tiếp và nguyên nhân của các ảnh lỗi sau khi xem kết quả.
Giới hạn đo lường: một seed, test nhỏ và tương quan video, threshold IoU 0.5, không đánh giá tốc độ triển khai thời gian thực.
"""
    if "e5" in names:
        stats = {n: hacking(entries, pred[n]) for n in (best, "e5")}
        dump("report/diagnostics/reward_hacking.json", stats)
        text += "\n## Bonus E5\n" + f"GRPO và SFT: {json.dumps(stats)}.\n"
        text += "Theo dõi tỷ lệ [] trên ảnh có nhãn, box khổng lồ, parse rate và FP/ảnh cùng F1/CI; reward tăng một mình chưa chứng minh grounding tốt hơn. Log: report/runs/e5_grpo_log.json.\n"
    Path("REPORT.generated.md").write_text(text, encoding="utf-8")
    print("Wrote REPORT.generated.md. Review findings and personal reflection before copying to REPORT.md.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    p.add_argument("--student", default=STUDENT)
    a = p.parse_args()
    preview() if a.preview else generate(a.student)
