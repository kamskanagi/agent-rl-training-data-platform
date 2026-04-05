from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_submit_feedback(client):
    # Create task and annotator first
    task_resp = await client.post("/api/tasks/", json={
        "prompt": "Test prompt",
        "responses": [{"text": "a"}, {"text": "b"}],
    })
    task_id = task_resp.json()["id"]

    ann_resp = await client.post("/api/annotators/", json={
        "email": "ann1@test.com",
        "name": "Ann One",
    })
    ann_id = ann_resp.json()["id"]

    resp = await client.post("/api/feedback/", json={
        "task_id": task_id,
        "annotator_id": ann_id,
        "ranking": [1, 2],
        "confidence": 0.9,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["ranking"] == [1, 2]


@pytest.mark.asyncio
async def test_feedback_recomputes_quality(client):
    task_resp = await client.post("/api/tasks/", json={
        "prompt": "Quality test",
        "responses": [{"text": "a"}, {"text": "b"}],
        "min_annotations": 2,
    })
    task_id = task_resp.json()["id"]

    # Create 2 annotators
    ann1 = (await client.post("/api/annotators/", json={
        "email": "q1@test.com", "name": "Q1",
    })).json()["id"]
    ann2 = (await client.post("/api/annotators/", json={
        "email": "q2@test.com", "name": "Q2",
    })).json()["id"]

    # Submit feedback from both
    await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann1,
        "ranking": [1, 2], "scalar_reward": 0.8,
    })
    await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann2,
        "ranking": [1, 2], "scalar_reward": 0.85,
    })

    # Task should now be completed with quality scores
    task = (await client.get(f"/api/tasks/{task_id}")).json()
    assert task["status"] == "completed"
    assert task["quality_score"] is not None
    assert task["iaa"] is not None


@pytest.mark.asyncio
async def test_get_task_feedback(client):
    task_resp = await client.post("/api/tasks/", json={"prompt": "Feedback list"})
    task_id = task_resp.json()["id"]
    ann_resp = await client.post("/api/annotators/", json={
        "email": "fl@test.com", "name": "FL",
    })
    ann_id = ann_resp.json()["id"]

    await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann_id,
        "scalar_reward": 0.7,
    })

    resp = await client.get(f"/api/feedback/task/{task_id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_flag_feedback(client):
    task_resp = await client.post("/api/tasks/", json={"prompt": "Flag fb"})
    task_id = task_resp.json()["id"]
    ann_resp = await client.post("/api/annotators/", json={
        "email": "ffb@test.com", "name": "FFB",
    })
    ann_id = ann_resp.json()["id"]

    fb_resp = await client.post("/api/feedback/", json={
        "task_id": task_id, "annotator_id": ann_id,
        "scalar_reward": 0.5,
    })
    fb_id = fb_resp.json()["id"]

    resp = await client.post(f"/api/feedback/{fb_id}/flag")
    assert resp.status_code == 200
    assert resp.json()["flagged"] is True
