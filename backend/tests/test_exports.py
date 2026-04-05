from __future__ import annotations

import pytest


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
