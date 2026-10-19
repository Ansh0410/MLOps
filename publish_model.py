
import os
import re
import json
import base64
import hashlib
from pathlib import Path
from urllib.parse import quote

import requests

OWNER = "Ansh0410"
REPO = "tourism-model-registry"
BRANCH = "main"

MODEL_PATH = Path(
    "artifacts/models/best_tourism_model.joblib"
)
MANIFEST_PATH = Path(
    "reports/model_manifest.json"
)

TOKEN = os.environ.get("MODEL_REGISTRY_TOKEN")

if not TOKEN:
    raise RuntimeError("MODEL_REGISTRY_TOKEN is missing.")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}


def sha256_file(path):
    """Calculate SHA-256 fingerprint for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_file(source_path, destination):
    """
    Publish a file to the private GitHub repository.

    If the destination already exists with identical
    content, skip it.

    If it exists with different content, stop instead
    of overwriting an established model version.
    """
    content = source_path.read_bytes()

    # GitHub Contents API has a 100 MB file limit.
    if len(content) > 100 * 1024 * 1024:
        raise RuntimeError("File exceeds GitHub API limit.")

    url = (
        f"https://api.github.com/repos/"
        f"{OWNER}/{REPO}/contents/"
        f"{quote(destination, safe='/')}"
    )

    existing = requests.get(
        url,
        headers=HEADERS,
        params={"ref": BRANCH},
        timeout=30
    )

    if existing.status_code == 200:
        # Git blob SHA compares exact file contents.
        existing_sha = existing.json()["sha"]

        current_sha = hashlib.sha1(
            b"blob " +
            str(len(content)).encode() +
            b"\0" + content
        ).hexdigest()

        if existing_sha == current_sha:
            print("ALREADY STORED:", destination)
            return

        raise RuntimeError(
            f"Version conflict: {destination}"
        )

    if existing.status_code != 404:
        existing.raise_for_status()

    payload = {
        "message": f"Publish model artifact: {destination}",
        "content": base64.b64encode(
            content
        ).decode("utf-8"),
        "branch": BRANCH
    }

    result = requests.put(
        url,
        headers=HEADERS,
        json=payload,
        timeout=120
    )

    if result.status_code != 201:
        result.raise_for_status()
        raise RuntimeError("Unexpected publish response.")

    print("PUBLISHED:", destination)


def main():
    # Verify necessary files exist.
    for path in [MODEL_PATH, MANIFEST_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Load the model's recorded provenance.
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    version = manifest["version_id"]

    # Ensure the version is safe for repository paths.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", version):
        raise ValueError("Invalid model version identifier.")

    # Verify actual model bytes against the manifest.
    actual_sha = sha256_file(MODEL_PATH)

    if actual_sha != manifest["model_sha256"]:
        raise RuntimeError(
            "Model SHA-256 does not match the manifest."
        )

    # Ensure we publish the model produced by this run.
    if manifest["source_git_commit"] != os.environ["GITHUB_SHA"]:
        raise RuntimeError("Git commit mismatch.")

    if str(manifest["github_run_id"]) != os.environ["GITHUB_RUN_ID"]:
        raise RuntimeError("GitHub run ID mismatch.")

    destination = f"models/{version}"

    print("Publishing model version:", version)

    # Store model first, then its manifest.
    publish_file(
        MODEL_PATH,
        f"{destination}/best_tourism_model.joblib"
    )

    publish_file(
        MANIFEST_PATH,
        f"{destination}/model_manifest.json"
    )

    print("Private model publication completed.")


if __name__ == "__main__":
    main()
