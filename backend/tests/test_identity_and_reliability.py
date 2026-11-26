from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.scoring import annotator_agreement, pairwise_agreement, reliability_from_samples
from tests.conftest import register

RESPONSES = [{"model_id": "a", "text": "A"}, {"model_id": "b", "text": "B"}, {"model_id": "c", "text": "C"}]


async def _task(client, min_annotations=3):
    return (await client.post("/api/tasks/", json={
        "prompt": "p", "responses": RESPONSES, "min_annotations": min_annotations,
    })).json()["id"]


async def _submit(client, task_id, headers, **body):
    return await client.post("/api/feedback/", json={"task_id": task_id, **body}, headers=headers)


# --- Identity ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_returned_once_and_never_listed(client):
    resp = await client.post("/api/annotators/", json={"email": "t@test.com", "name": "T"})
    assert resp.status_code == 201
    assert len(resp.json()["api_token"]) > 20

    listed = (await client.get("/api/annotators/")).json()
    assert all("api_token" not in a and "api_token_hash" not in a for a in listed)
    single = (await client.get(f"/api/annotators/{resp.json()['id']}")).json()
    assert "api_token" not in single and "api_token_hash" not in single


@pytest.mark.asyncio
async def test_me_resolves_token(client):
    ann_id, headers = await register(client, "me@test.com")
    resp = await client.get("/api/annotators/me", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == ann_id


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-real-token"}])
async def test_feedback_requires_valid_token(client, headers):
    task_id = await _task(client)
    resp = await _submit(client, task_id, headers, ranking=[1, 2, 3])
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_cannot_submit_as_someone_else(client):
    victim_id, _ = await register(client, "victim@test.com")
    _, attacker_headers = await register(client, "attacker@test.com")
    task_id = await _task(client)

    resp = await _submit(client, task_id, attacker_headers, annotator_id=victim_id, ranking=[1, 2, 3])
    assert resp.status_code == 403
    assert (await client.get(f"/api/feedback/task/{task_id}")).json() == []


@pytest.mark.asyncio
async def test_annotator_taken_from_token(client):
    ann_id, headers = await register(client, "who@test.com")
    task_id = await _task(client)
    resp = await _submit(client, task_id, headers, ranking=[1, 2, 3])
    assert resp.status_code == 201
    assert resp.json()["annotator_id"] == ann_id


@pytest.mark.asyncio
async def test_next_task_only_for_yourself(client):
    other_id, _ = await register(client, "other@test.com")
    _, headers = await register(client, "me2@test.com")
    await _task(client)
    resp = await client.get(f"/api/annotators/{other_id}/next-task", headers=headers)
    assert resp.status_code == 403


# --- Duplicates and validation -----------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_feedback_rejected(client):
    _, headers = await register(client, "dup@test.com")
    task_id = await _task(client)
    assert (await _submit(client, task_id, headers, ranking=[1, 2, 3])).status_code == 201
    resp = await _submit(client, task_id, headers, ranking=[3, 2, 1])
    assert resp.status_code == 409
    assert len((await client.get(f"/api/feedback/task/{task_id}")).json()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ranking", [[0, 1, 2], [1, 1, 2], [1, 2], [2, 0, 1], [1, 2, 3, 4]])
async def test_malformed_ranking_rejected(client, ranking):
    _, headers = await register(client, "rank@test.com")
    task_id = await _task(client)
    resp = await _submit(client, task_id, headers, ranking=ranking)
    assert resp.status_code == 422


# --- Queue ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_requeued_until_enough_annotations(client):
    task_id = await _task(client, min_annotations=2)
    a_id, a_headers = await register(client, "qa@test.com")
    b_id, b_headers = await register(client, "qb@test.com")

    got = await client.get(f"/api/annotators/{a_id}/next-task", headers=a_headers)
    assert got.json()["id"] == task_id
    await _submit(client, task_id, a_headers, ranking=[1, 2, 3])

    # A has done it, so A is not offered it again; B is.
    assert (await client.get(f"/api/annotators/{a_id}/next-task", headers=a_headers)).status_code == 404
    got = await client.get(f"/api/annotators/{b_id}/next-task", headers=b_headers)
    assert got.json()["id"] == task_id
    await _submit(client, task_id, b_headers, ranking=[1, 2, 3])

    # Completed: no longer queued.
    assert (await client.get(f"/api/annotators/{b_id}/next-task", headers=b_headers)).status_code == 404


# --- Reliability ---------------------------------------------------------------

def fb(annotator_id, task_id="t", ranking=None, scalar_reward=None, flagged=False):
    return SimpleNamespace(annotator_id=annotator_id, task_id=task_id, ranking=ranking,
                           scalar_reward=scalar_reward, binary_label=None, flagged=flagged)


def test_pairwise_agreement():
    assert pairwise_agreement(fb("a", ranking=[1, 2, 3]), fb("b", ranking=[1, 2, 3])) == 1.0
    assert pairwise_agreement(fb("a", ranking=[1, 2, 3]), fb("b", ranking=[3, 2, 1])) == 0.0
    assert pairwise_agreement(fb("a", ranking=[1, 2, 3]), fb("b", ranking=[1, 3, 2])) == pytest.approx(2 / 3)
    assert pairwise_agreement(fb("a", scalar_reward=0.9), fb("b", scalar_reward=0.6)) == pytest.approx(0.7)
    assert pairwise_agreement(fb("a", ranking=[1, 2]), fb("b", scalar_reward=0.5)) is None


def test_agreement_samples_and_flagging():
    feedbacks = [
        fb("a", "t1", [1, 2, 3]), fb("b", "t1", [1, 2, 3]), fb("c", "t1", [3, 2, 1]),
        fb("a", "t2", [1, 2, 3], flagged=True), fb("b", "t2", [1, 2, 3]),
        fb("a", "t3", [1, 2, 3]),  # alone on the task: no sample
    ]
    assert annotator_agreement(feedbacks, "a") == [0.5, 0.0]
    assert annotator_agreement(feedbacks, "c") == [0.0]


def test_reliability_shrinks_toward_one():
    assert reliability_from_samples([]) == 1.0
    assert reliability_from_samples([0.0]) == pytest.approx(0.75)
    assert reliability_from_samples([0.0] * 30) < 0.1


@pytest.mark.asyncio
async def test_reliability_updates_from_real_agreement(client):
    ids = {}
    headers = {}
    for name in ("good1", "good2", "contrarian"):
        ids[name], headers[name] = await register(client, f"{name}@test.com")

    for _ in range(3):
        task_id = await _task(client)
        await _submit(client, task_id, headers["good1"], ranking=[1, 2, 3])
        await _submit(client, task_id, headers["good2"], ranking=[1, 2, 3])
        await _submit(client, task_id, headers["contrarian"], ranking=[3, 2, 1])

    get = lambda n: client.get(f"/api/annotators/{ids[n]}")
    good, bad = (await get("good1")).json(), (await get("contrarian")).json()
    # good1 agrees fully with good2 and not at all with contrarian: 0.5 per task.
    assert good["avg_agreement_rate"] == pytest.approx(0.5)
    assert bad["avg_agreement_rate"] == 0.0
    assert good["reliability_score"] > bad["reliability_score"]
    assert bad["reliability_score"] == pytest.approx(3 / 6)


@pytest.mark.asyncio
async def test_flagging_lowers_reliability(client):
    a_id, a_headers = await register(client, "fa@test.com")
    _, b_headers = await register(client, "fb@test.com")
    task_id = await _task(client)
    fb_a = (await _submit(client, task_id, a_headers, ranking=[1, 2, 3])).json()
    await _submit(client, task_id, b_headers, ranking=[1, 2, 3])
    before = (await client.get(f"/api/annotators/{a_id}")).json()["reliability_score"]

    await client.post(f"/api/feedback/{fb_a['id']}/flag")
    after = (await client.get(f"/api/annotators/{a_id}")).json()["reliability_score"]
    assert before == 1.0 and after == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_requeue_does_not_duplicate(client, annotation_queue):
    task_id = await _task(client, min_annotations=3)  # queued once on creation
    for name in ("d1", "d2"):
        _, headers = await register(client, f"{name}@test.com")
        await _submit(client, task_id, headers, ranking=[1, 2, 3])  # without pulling it first
    assert annotation_queue.count(task_id) == 1
