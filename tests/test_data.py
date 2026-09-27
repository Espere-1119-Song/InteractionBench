"""Dataset loader: public layout, earlier layout, counting tags."""

import json

from interactionbench import iter_items, load_benchmark


def test_public_layout(bench):
    videos = load_benchmark(bench)
    assert [v.video_id for v in videos] == ["vidAAAAAAAA", "vidBBBBBBBB"]
    items = list(iter_items(videos))
    assert [it.item_id for it in items] == ["vidAAAAAAAA#0", "vidAAAAAAAA#1",
                                            "vidBBBBBBBB#0", "vidBBBBBBBB#1"]
    assert items[0].domain == "demo" and items[0].duration_s == 20.0
    assert items[2].is_counting and items[3].should_remain_silent
    assert load_benchmark(bench / "annotations")[0].video_id == "vidAAAAAAAA"


def test_earlier_layout_and_tags(tmp_path):
    d = tmp_path / "results" / "demo" / "vidCCCCCCCC"
    d.mkdir(parents=True)
    (d / "annotation.json").write_text(json.dumps({
        "category": "demo", "video_id": "vidCCCCCCCC", "duration_s": 9, "annotator": "x",
        "items": [
            {"capability": "CST", "time_type": "C", "sub_tag": "\u8ba1\u6570\u578b",
             "question": "q", "question_time_s": 0, "answers": [{"time_s": 1, "content": "1"}]},
            {"capability": "CST", "time_type": "C", "sub_tag": "counting",
             "question": "q", "question_time_s": 0, "answers": [{"time_s": 1, "content": "1"}]},
            {"capability": "LCG", "time_type": "C", "sub_tag": "narration",
             "question": "q", "question_time_s": 0, "answers": [{"time_s": 1, "content": "a"}]},
        ]}))
    items = list(iter_items(load_benchmark(tmp_path)))
    assert [it.is_counting for it in items] == [True, True, False]
    assert items[0].domain == "demo" and items[0].item_id == "vidCCCCCCCC#0"
