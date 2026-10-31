"""Hybrid classifier (lexicon + learned model). A tiny model trained on ARTIFICIAL texts keeps these tests hermetic:
the real model (make train) is built from confidential tickets and is never needed here."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("sklearn")

from app import classify  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.ml import ComplaintModel, train  # noqa: E402
from app.pipeline import TriageService  # noqa: E402
from tests.conftest import make_req  # noqa: E402

TOY = {
    "complaint.admin": ["не согласен с наказанием снимите бан", "дали мут не за что прошу снять", "бан выдан ошибочно я не нарушал",
                        "наказание несправедливое пересмотрите", "снимите варн я ничего не делал"],
    "complaint.insult": ["оскорбил меня в чате", "оск родни в чате", "оскорблял мою семью", "мат в чате оскорбления", "оскорбил маму"],
    "complaint.dm": ["убил без причины у банка", "дм в зеленой зоне", "убил меня просто так", "стрелял без причины", "дб на машине"],
    "complaint.cheat": ["летает по воздуху читы", "аимбот стреляет через стены", "использует софт", "читер телепортируется", "спидхак"],
}


LEXICON_ONLY = pytest.mark.skipif(os.environ.get("TEST_USE_MODEL") == "1", reason="asserts the hermetic lexicon-only mode")


@pytest.fixture(scope="module")
def toy_model() -> ComplaintModel:
    texts, labels = [], []
    for cat, xs in TOY.items():
        for x in xs:
            for _ in range(3):  # min_df=3 of the char vectorizer needs repetitions in such a small corpus
                texts.append(x)
                labels.append(cat)
    return train(texts, labels, meta={"n_train": len(texts)})


@pytest.fixture()
def hybrid(rulebook, tmp_path, toy_model) -> TriageService:
    os.environ["DB_PATH"] = str(tmp_path / "t.db")
    return TriageService(load_settings(), rules=rulebook, model=toy_model)


def test_model_probabilities_and_roundtrip(toy_model, tmp_path):
    p = toy_model.proba("снимите бан не согласен")
    assert set(p) == set(TOY) and abs(sum(p.values()) - 1) < 1e-6
    assert max(p, key=p.get) == "complaint.admin"
    path = tmp_path / "m.joblib"
    toy_model.save(path)
    again = ComplaintModel.load(path)
    assert again is not None and again.classes == toy_model.classes and again.known_words == toy_model.known_words
    assert again.proba("оскорбил меня") == pytest.approx(toy_model.proba("оскорбил меня"))


def test_missing_or_broken_model_is_never_fatal(tmp_path, capsys):
    assert ComplaintModel.load(tmp_path / "absent.joblib") is None
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    assert ComplaintModel.load(bad) is None
    assert "lexicon only" in capsys.readouterr().err


@LEXICON_ONLY
def test_service_without_model_reports_lexicon_only(service):
    assert service.model is None and "lexicon only" in service.classifier


def test_hybrid_is_explained_in_reasons(hybrid):
    r, dbg = hybrid.triage(make_req("Arman_Petrov оскорбил меня в чате"))
    assert r.category == "complaint.insult"
    assert any("модель" in x for x in r.reasons)
    assert dbg.signals["model_proba"]["complaint.insult"] > 0.3
    assert "hybrid" in hybrid.classifier


def test_model_corrects_an_appeal_the_lexicon_calls_dm(service, hybrid, toy_model):
    text = "снимите наказание за дм, бан выдан ошибочно"
    assert hybrid.triage(make_req(text))[0].category == "complaint.admin"
    assert hybrid.triage(make_req(text))[0].needs_human is True


def test_invalid_and_safety_stay_rule_based(hybrid):
    r, _ = hybrid.triage(make_req("жб"))
    assert r.category == "complaint.invalid"  # the model has no `invalid` class; structure decides
    r, _ = hybrid.triage(make_req("он угрожает убить меня в реальной жизни, оскорбил в чате"))
    assert r.priority == "p0" and r.needs_human
    r, _ = hybrid.triage(make_req("оплатил донат, коины не пришли", kind="report", att=0))
    assert r.category == "report.payment" and r.priority == "p0" and r.needs_human


def test_low_vocabulary_coverage_lets_the_lexicon_decide(toy_model):
    # every word is unseen by the toy model: its weight shrinks and a clear lexicon match wins
    text = "нақты ақшаға аккаунт сатады"
    assert toy_model.coverage(text) < 0.5
    a = classify.analyze(make_req(text), toy_model)
    assert a.category == "complaint.rmt_ads"


def test_blend_weight_bounds():
    scores = {"complaint.insult": 6.0, "complaint.dm": 0.0}
    proba = {"complaint.insult": 0.1, "complaint.dm": 0.9}
    full = classify._blend(scores, proba, coverage=1.0)
    none = classify._blend(scores, proba, coverage=0.0)
    assert full["complaint.dm"] > none["complaint.dm"] and none["complaint.insult"] == pytest.approx(1.0)
    empty = classify._blend({"complaint.insult": 0.0, "complaint.dm": 0.0}, proba, coverage=0.0)
    assert max(empty, key=empty.get) == "complaint.dm"  # nothing in the lexicon: the model decides
