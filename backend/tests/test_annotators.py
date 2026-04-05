from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_annotator(client):
    resp = await client.post("/api/annotators/", json={
        "email": "alice@test.com",
        "name": "Alice",
        "role": "senior",
        "expertise_tags": ["python", "ml"],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "alice@test.com"
    assert data["reliability_score"] == 1.0


@pytest.mark.asyncio
async def test_duplicate_email(client):
    await client.post("/api/annotators/", json={
        "email": "dup@test.com", "name": "First",
    })
    resp = await client.post("/api/annotators/", json={
        "email": "dup@test.com", "name": "Second",
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_annotators(client):
    await client.post("/api/annotators/", json={
        "email": "a1@test.com", "name": "A1",
    })
    await client.post("/api/annotators/", json={
        "email": "a2@test.com", "name": "A2",
    })
    resp = await client.get("/api/annotators/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_annotator(client):
    create_resp = await client.post("/api/annotators/", json={
        "email": "get@test.com", "name": "Get",
    })
    ann_id = create_resp.json()["id"]
    resp = await client.get(f"/api/annotators/{ann_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ann_id


@pytest.mark.asyncio
async def test_next_task(client):
    # Create task (which enqueues to mock queue)
    task_resp = await client.post("/api/tasks/", json={"prompt": "Queue task"})
    task_id = task_resp.json()["id"]

    ann_resp = await client.post("/api/annotators/", json={
        "email": "next@test.com", "name": "Next",
    })
    ann_id = ann_resp.json()["id"]

    resp = await client.get(f"/api/annotators/{ann_id}/next-task")
    assert resp.status_code == 200
    assert resp.json()["id"] == task_id


@pytest.mark.asyncio
async def test_next_task_empty_queue(client):
    ann_resp = await client.post("/api/annotators/", json={
        "email": "empty@test.com", "name": "Empty",
    })
    ann_id = ann_resp.json()["id"]

    resp = await client.get(f"/api/annotators/{ann_id}/next-task")
    assert resp.status_code == 404
