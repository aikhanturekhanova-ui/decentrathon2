"""Reproduce the "preliminary checks" of the data (aggregates only) -> docs/metrics/data_profile.json."""
from __future__ import annotations

import collections
import json

from _common import DATA, METRICS, load_jsonl, player_text
from app.knowledge import RuleBook
from app.textutil import GAME_CHARS, KK_LETTERS, detect_lang


def main() -> None:
    train = load_jsonl(DATA / "tickets_train.jsonl.gz")
    valid = load_jsonl(DATA / "tickets_valid.jsonl")
    rb = RuleBook.load(DATA / "rules.json")
    taxonomy = json.loads((DATA / "taxonomy.json").read_text(encoding="utf-8"))

    def stats(d):
        texts = [player_text(t) for t in d]
        return {
            "n": len(d), "kind": dict(collections.Counter(t["kind"] for t in d)),
            "dataset_lang_label": dict(collections.Counter(t["lang"] for t in d).most_common()),
            "server_lang": dict(collections.Counter(t.get("server_lang") for t in d).most_common()),
            "verdict": dict(collections.Counter(t.get("verdict") for t in d).most_common()),
            "has_category_or_priority": any("category" in t or "priority" in t for t in d),
            "images_count_min_for_complaints": min((t["images_count"] for t in d if t["kind"] == "complaint"), default=None),
            "player_text_len_median": sorted(len(x) for x in texts)[len(texts) // 2],
            "texts_with_game_symbols": sum(any(c in GAME_CHARS for c in x) for x in texts),
            "texts_with_true_kazakh_letters": sum(any(c in KK_LETTERS for c in x) for x in texts),
        }

    sample = valid
    ours_vs_label = collections.Counter((t["lang"], detect_lang(player_text(t), t.get("server_lang"))) for t in sample)
    ids_train = {t["ticket_id"] for t in train}
    rep_train = {t["reporter_hash"] for t in train}
    trtexts = {player_text(t).lower() for t in train}
    dup_refs = collections.defaultdict(list)
    for d in rb.docs:
        dup_refs[d.ref].append(d.section_id)
    out = {
        "train": stats(train), "valid": stats(valid),
        "valid_overlap_with_train": {
            "ticket_ids": sum(t["ticket_id"] in ids_train for t in valid),
            "reporters": sum(t["reporter_hash"] in rep_train for t in valid),
            "exact_player_text": sum(player_text(t).lower() in trtexts for t in valid),
        },
        "lang_label_vs_our_detector_on_valid": {f"{a}->{b}": n for (a, b), n in ours_vs_label.most_common()},
        "rules": {"sections": len({d.section_id for d in rb.docs}), "points": len(rb.docs), "headings": sum(d.is_heading for d in rb.docs),
                  "sections_without_points_but_with_text": len({d.section_id for d in rb.context_docs}),
                  "refs_used_in_more_than_one_section": sum(len(set(v)) > 1 for v in dup_refs.values())},
        "taxonomy_version": taxonomy.get("version"), "taxonomy_note": taxonomy.get("note"),
    }
    (METRICS / "data_profile.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("valid_overlap_with_train", "rules")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
