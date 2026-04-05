from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_task(client):
    resp = await client.post("/api/tasks/", json={
        "prompt": "Write a Python function to sort a list",
        "responses": [
            {"model_id": "m-a", "text": "def sort(lst): return sorted(lst)"},
            {"model_id": "m-b", "text": "def sort(lst): lst.sort(); return lst"},
        ],
        "annotation_type": "ranking",
        "min_annotations": 3,
        "tags": ["python", "sorting"],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["prompt"] == "Write a Python function to sort a list"
    assert data["status"] == "pending"
    assert data["annotation_type"] == "ranking"


@pytest.mark.asyncio
async def test_list_tasks(client):
    await client.post("/api/tasks/", json={"prompt": "Task 1"})
    await client.post("/api/tasks/", json={"prompt": "Task 2"})
    resp = await client.get("/api/tasks/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["tasks"]) == 2


@pytest.mark.asyncio
async def test_get_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Get me"})
    task_id = create_resp.json()["id"]
    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == task_id


@pytest.mark.asyncio
async def test_get_task_not_found(client):
    resp = await client.get("/api/tasks/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Original"})
    task_id = create_resp.json()["id"]
    resp = await client.patch(f"/api/tasks/{task_id}", json={"prompt": "Updated"})
    assert resp.status_code == 200
    assert resp.json()["prompt"] == "Updated"


@pytest.mark.asyncio
async def test_delete_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Delete me"})
    task_id = create_resp.json()["id"]
    resp = await client.delete(f"/api/tasks/{task_id}")
    assert resp.status_code == 204
    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_flag_task(client):
    create_resp = await client.post("/api/tasks/", json={"prompt": "Flag me"})
    task_id = create_resp.json()["id"]
    resp = await client.post(f"/api/tasks/{task_id}/flag")
    assert resp.status_code == 200
    assert resp.json()["status"] == "flagged"


@pytest.mark.asyncio
async def test_filter_tasks_by_status(client):
    await client.post("/api/tasks/", json={"prompt": "Task 1"})
    create2 = await client.post("/api/tasks/", json={"prompt": "Task 2"})
    task_id = create2.json()["id"]
    await client.post(f"/api/tasks/{task_id}/flag")

    resp = await client.get("/api/tasks/?status=flagged")
    data = resp.json()
    assert data["total"] == 1
    assert data["tasks"][0]["status"] == "flagged"
