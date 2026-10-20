
import base64
import hashlib
import json
import os

import joblib
import requests

BASE_URL = (
    "https://api.github.com/repos/"
    "Ansh0410/tourism-model-registry"
)


def get_file(path, token):
    """Download a file from the private model repository."""

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    response = requests.get(
        f"{BASE_URL}/contents/{path}",
        headers=headers,
        params={"ref": "main"},
        timeout=60
    )
    response.raise_for_status()

    metadata = response.json()

    if metadata.get("encoding") == "base64":
        return base64.b64decode(metadata["content"])

    # Fallback for larger files without inline content.
    blob_sha = metadata.get("sha")

    if not blob_sha:
        raise RuntimeError("GitHub did not provide file content.")

    blob_response = requests.get(
        f"{BASE_URL}/git/blobs/{blob_sha}",
        headers=headers,
        timeout=120
    )
    blob_response.raise_for_status()

    blob = blob_response.json()

    if blob.get("encoding") != "base64":
        raise RuntimeError("Unsupported GitHub blob encoding.")

    return base64.b64decode(blob["content"])


def load_approved_model(token):
    """Load the staging-approved model after integrity checks."""

    if not token:
        raise RuntimeError("Model registry access token is missing.")

    # 1. Identify the approved version.
    staging = json.loads(
        get_file("deployment/staging.json", token)
    )

    if staging.get("status") != "approved_for_staging":
        raise RuntimeError("No model approved for staging.")

    # 2. Read the corresponding model manifest.
    manifest = json.loads(
        get_file(staging["manifest_path"], token)
    )

    if manifest["version_id"] != staging["model_version"]:
        raise RuntimeError("Model version mismatch.")

    # 3. Retrieve the exact approved model.
    model_bytes = get_file(staging["model_path"], token)

    # 4. Verify SHA-256 before deserialization.
    actual_hash = hashlib.sha256(model_bytes).hexdigest()

    if actual_hash != manifest["model_sha256"]:
        raise RuntimeError("Model hash mismatch with manifest.")

    if actual_hash != staging["model_sha256"]:
        raise RuntimeError("Model hash mismatch with approval.")

    # 5. Check recorded Scikit-learn version if available.
    import sklearn
    expected_version = manifest.get(
        "environment", {}
    ).get("scikit_learn")

    if expected_version and expected_version != sklearn.__version__:
        raise RuntimeError(
            f"Scikit-learn mismatch: model={expected_version}, "
            f"runtime={sklearn.__version__}"
        )

    # 6. Deserialize only after verifying the trusted artifact.
    # Use temporary local storage for joblib.
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from sklearn.exceptions import InconsistentVersionWarning
    import warnings

    with TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.joblib"
        model_path.write_bytes(model_bytes)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", InconsistentVersionWarning
            )
            model = joblib.load(model_path)

    return model, staging["model_version"]
