import json
from pathlib import Path
from PIL import Image
from scripts.score import evaluate
from student.common import dump, objects_to_answer
from student.reporting import generate
from student.run import score

def test_report_uses_real_evidence_and_leaves_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"data/images").mkdir(parents=True)
    entries = []
    for i in range(8):
        image_path = f"images/{i}.jpg"
        Image.new("RGB",(640,360),"gray").save(tmp_path/"data"/image_path)
        entries.append({"image_id":str(i),"image_path":image_path,"width":640,"height":360,
                        "session":f"s{i//2}", "objects":[{"label":c,"yolo":[.2+j*.2,.3,.1,.1]}
                        for j,c in enumerate(("pothole","crack","manhole"))]})
    (tmp_path/"data/splits").mkdir()
    (tmp_path/"data/splits/test.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
    dump("data/splits/meta.json",{"dataset_version":"d10-fixture"})
    dump("report/selection.json",{"best":"e1"})
    dump("report/diagnostics/data_audit.json",{"splits":{}})
    dump("report/diagnostics/mask_audit.json",{"image_tokens":220,"total_tokens":350,
         "supervised_tokens":100,"supervised_text":"[]<|im_end|>"})
    Path("REPORT.md").write_text("keep original")
    resource = {"trainable_params":12,"peak_vram_gb":1.2,"train_minutes":3.4}
    for name in ("e0","e1","e2","e3","e4_data_size_0.1","e4_data_size_1.0"):
        pred = {e["image_id"]:objects_to_answer(e["objects"]) if int(e["image_id"]) < 4 else "[]"
                for e in entries}
        result = evaluate(entries,pred)
        result.update(dataset_version="d10-fixture",split="test",meta=resource,score_version=1)
        dump(f"report/evidence/{name}.json",result)
        Path("work").mkdir(exist_ok=True)
        Path(f"work/{name}_test.jsonl").write_text("\n".join(
            json.dumps({"image_id":k,"raw":v}) for k,v in pred.items()))
        if name in ("e1","e2","e3"):
            dump(f"report/runs/{name}.json",{**resource,"architecture":{"projector":100},
                                          "matched_modules":["example"]})
    generate("Test Student")
    text = Path("REPORT.generated.md").read_text(encoding="utf-8")
    assert "Test Student" in text and "d10-fixture" in text
    assert text.count("### Câu ") == 10
    assert len(list(Path("report/figures").glob("prediction_*.png"))) == 6
    assert Path("REPORT.md").read_text() == "keep original"
    assert json.loads(Path("report/diagnostics/e1_wrong_pixel_scale.json").read_text())["micro"]["f1"] == 0
