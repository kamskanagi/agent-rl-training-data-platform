from __future__ import annotations

from unittest.mock import patch

import pytest

from routes.exports import _build_export as real_build_export
from tests.conftest import register


@pytest.mark.asyncio
async def test_create_dataset(client):
    resp = await client.post("/api/exports/datasets", json={
        "name": "test-export",
        "filters": {"min_quality": 0.5},
        "export_format": "jsonl",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-export"
    assert data["export_format"] == "jsonl"


@pytest.mark.asyncio
async def test_list_datasets(client):
    await client.post("/api/exports/datasets", json={"name": "ds1"})
    await client.post("/api/exports/datasets", json={"name": "ds2"})
    resp = await client.get("/api/exports/datasets")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_get_dataset(client):
    create_resp = await client.post("/api/exports/datasets", json={"name": "get-ds"})
    ds_id = create_resp.json()["id"]
    resp = await client.get(f"/api/exports/datasets/{ds_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ds_id


@pytest.mark.asyncio
async def test_download_not_ready(client):
    create_resp = await client.post("/api/exports/datasets", json={"name": "dl-ds"})
    ds_id = create_resp.json()["id"]
    resp = await client.get(f"/api/exports/datasets/{ds_id}/download")
    assert resp.status_code == 404


async def _completed_task(client, prompt, rewards):
    """Create a 2-response task and complete it with one scalar reward per annotator."""
    task = (await client.post("/api/tasks/", json={
        "prompt": prompt,
        "responses": [{"model_id": "m-a", "text": "a"}, {"model_id": "m-b", "text": "b"}],
        "min_annotations": len(rewards),
    })).json()
    for i, reward in enumerate(rewards):
        _, headers = await register(client, f"{prompt}-{i}@test.com")
        await client.post("/api/feedback/", json={
            "task_id": task["id"], "scalar_reward": reward,
        }, headers=headers)
    return (await client.get(f"/api/tasks/{task['id']}")).json()


@pytest.mark.asyncio
async def test_export_builds_via_api_and_applies_quality_filter(client, tmp_path):
    """Regression: the background builder used to run before the dataset row was
    committed, so exports created through the API were never built. Also checks that
    the "min_quality_score" key sent by the Exports page is honoured."""
    from tests.conftest import test_session

    high = await _completed_task(client, "high", [0.8, 0.8])  # no spread -> high quality
    low = await _completed_task(client, "low", [0.0, 1.0])    # max spread -> low quality
    assert high["status"] == low["status"] == "completed"
    assert high["quality_score"] > 0.5 > low["quality_score"]  # 0.6 vs 0.4

    with patch("routes.exports._build_export", real_build_export), \
         patch("routes.exports.async_session", test_session), \
         patch("routes.exports.EXPORT_DIR", str(tmp_path)):
        resp = await client.post("/api/exports/datasets", json={
            "name": "filtered", "export_format": "jsonl",
            "filters": {"min_quality_score": 0.5},
        })
    assert resp.status_code == 201

    dataset = (await client.get(f"/api/exports/datasets/{resp.json()['id']}")).json()
    assert dataset["export_path"] is not None, "export was never built"
    assert dataset["task_count"] == 1

    download = await client.get(f"/api/exports/datasets/{dataset['id']}/download")
    assert download.status_code == 200
    assert '"prompt": "high"' in download.text
