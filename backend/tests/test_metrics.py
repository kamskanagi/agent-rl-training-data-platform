from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_platform_metrics(client):
    # Create some data
    await client.post("/api/tasks/", json={"prompt": "Metric task"})
    await client.post("/api/annotators/", json={
        "email": "met@test.com", "name": "Met",
    })

    resp = await client.get("/api/metrics/platform")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tasks"] == 1
    assert data["total_annotators"] == 1
    assert data["pending_tasks"] == 1


@pytest.mark.asyncio
async def test_platform_metrics_cache(client):
    """Second call should hit cache."""
    await client.get("/api/metrics/platform")
    resp = await client.get("/api/metrics/platform")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_list_training_runs(client):
    resp = await client.get("/api/metrics/training")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_training_run_not_found(client):
    resp = await client.get("/api/metrics/training/nonexistent")
    assert resp.status_code == 404
