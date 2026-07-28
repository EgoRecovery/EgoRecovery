import json
import os
from typing import Any

import pandas as pd
import requests
from scaleapi import ScaleClient

REQUEST_TIMEOUT_S = 60


def get_tasks(project_name: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch all completed tasks for a project."""
    headers = {"accept": "application/json"}
    base_url = "https://api.scale.com/v1/tasks"

    next_token = None
    tasks: list[dict[str, Any]] = []

    while True:
        params = {  # TODO: fetch all tasks included non completed tasks after scale explained how this is done
            "project": project_name,
            "include_attachment_url": "true",
            "limit": 100,
        }
        if next_token:
            params["next_token"] = next_token

        response = requests.get(
            base_url,
            headers=headers,
            params=params,
            auth=(api_key, ""),
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()

        tasks.extend(data.get("docs", []))

        next_token = data.get("next_token")
        if not next_token:
            break

    return tasks


def get_episode_hash(task: dict[str, Any]) -> str:
    """Extract episode hash from task['params']['attachments'][0]."""
    attachment = task["params"]["attachments"][0]
    # Example:
    # s3://scale-sales-uploads/egoverse/2026-03-17-01-42-37-000000/2026-03-17-01-42-37-000000.mp4
    return attachment.rstrip("/").split("/")[-2]


def build_df_from_tasks(
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a lookup from episode_hash -> task."""

    df = pd.DataFrame(columns=["_ID", "STATUS", "S3_ATTACHMENT", "SEQUENCE_ID"])
    for task in tasks:
        attachments = task.get("params", {}).get("attachments", [])
        if not attachments:
            continue

        episode_hash = get_episode_hash(task)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "_ID": task["task_id"],
                            "STATUS": task["status"],
                            "S3_ATTACHMENT": attachments[0],
                            "SEQUENCE_ID": episode_hash,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return df


def download_scale_annotation(client: ScaleClient, tid: str, out_path: str):
    task = client.get_task(tid)
    url = task.response["annotations"]["url"]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = json.loads(resp.text.rstrip("\x00"))
    path = os.path.join(out_path, f"{tid}.json")
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def get_completed_tasks(project_name: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch all completed tasks for a project."""
    headers = {"accept": "application/json"}
    base_url = "https://api.scale.com/v1/tasks"

    next_token = None
    tasks: list[dict[str, Any]] = []

    while True:
        params = {  # TODO: fetch all tasks included non completed tasks after scale explained how this is done
            "status": "completed",
            "project": project_name,
            "include_attachment_url": "true",
            "limit": 100,
        }
        if next_token:
            params["next_token"] = next_token

        response = requests.get(
            base_url,
            headers=headers,
            params=params,
            auth=(api_key, ""),
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()

        tasks.extend(data.get("docs", []))

        next_token = data.get("next_token")
        if not next_token:
            break

    return tasks
