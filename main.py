from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def cli() -> None:
    """CLI entrypoint for modelretrieval utilities."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _extract_unsplash_photo_id(photo_page_url: str) -> str:
    # Handles URLs such as: https://unsplash.com/photos/description-<photo_id>
    match = re.search(r"/photos/(?:[^/?#]+-)?([A-Za-z0-9_-]+)", photo_page_url)
    if not match:
        raise ValueError(f"Could not extract Unsplash photo id from URL: {photo_page_url}")
    return match.group(1)


def _get_json(url: str, headers: Optional[dict[str, str]] = None, timeout: int = 60) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_bytes(url: str, headers: Optional[dict[str, str]] = None, timeout: int = 60) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def _append_client_id(url: str, access_key: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'client_id': access_key})}"


@dataclass
class DownloadSummary:
    downloaded: int
    failed: int


def _normalize_id(raw_id: object, width: int) -> str:
    # Convert IDs like 1, "1", "01", "1.0" into zero-padded strings.
    raw_str = str(raw_id).strip()
    if not raw_str:
        return ""

    try:
        numeric_id = int(float(raw_str))
    except ValueError:
        return ""

    if numeric_id < 0:
        return ""
    return str(numeric_id).zfill(width)


def _download_models_service(
    civitai_token: str,
    project_root: Path,
    echo,
    model_ids: Optional[list[str]] = None,
) -> DownloadSummary:
    csv_path = project_root / "data" / "models.csv"
    output_dir = project_root / "data" / "models"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(csv_path, dtype={"id": "string", "model_version_id": "string"})
    required_cols = {"id", "model_version_id"}
    missing_cols = required_cols - set(dataframe.columns)
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing_cols))}")

    valid_ids = dataframe["id"].dropna().astype(str).str.strip()
    if valid_ids.empty:
        raise ValueError("No valid model ids found in data/models.csv.")
    max_id = valid_ids.map(lambda x: int(float(x))).max()
    width = max(2, len(str(max_id)))

    if model_ids:
        model_ids_set = {_normalize_id(raw_id, width) for raw_id in model_ids}
        model_ids_set.discard("")
        dataframe = dataframe[dataframe["id"].apply(lambda x: _normalize_id(x, width) in model_ids_set)]
        if dataframe.empty:
            raise ValueError("No matching model ids found in the CSV file.")

    downloaded = 0
    failed = 0

    for row_index, row in dataframe.iterrows():
        raw_id = row.get("id")
        raw_model_version_id = row.get("model_version_id")

        if pd.isna(raw_id) or pd.isna(raw_model_version_id):
            echo(f"[{row_index + 1}] Missing id or model_version_id, skipping")
            failed += 1
            continue

        model_id = _normalize_id(raw_id, width)
        model_version_id = str(raw_model_version_id).strip()
        if not model_id or not model_version_id:
            echo(f"[{row_index + 1}] Missing id or model_version_id, skipping")
            failed += 1
            continue

        params = urlencode({"token": civitai_token})
        url = f"https://civitai.com/api/download/models/{model_version_id}?{params}"
        target_path = output_dir / f"{model_id}.safetensors"

        try:
            request = Request(url, headers={"User-Agent": "modelretrieval-cli/0.1"})
            with urlopen(request, timeout=120) as response:
                content = response.read()
            target_path.write_bytes(content)
            downloaded += 1
            echo(f"[{row_index + 1}] Downloaded {target_path.name}")
        except (HTTPError, URLError, OSError) as error:
            failed += 1
            echo(f"[{row_index + 1}] Failed {model_version_id}: {error}")

    return DownloadSummary(downloaded=downloaded, failed=failed)


@app.command(name="download_models")
def download_models(
    civitai_token: str = typer.Option(..., "--civitai-token", help="Civitai API token"),
    model_ids: Optional[list[str]] = typer.Option(
        None,
        "--model-id",
        help="Model id from data/models.csv",
        show_default=False,
    ),
) -> None:
    """Download model files from Civitai into data/models/."""
    project_root = _repo_root()
    try:
        summary = _download_models_service(
            civitai_token=civitai_token,
            project_root=project_root,
            echo=typer.echo,
            model_ids=model_ids,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Done. Downloaded: {summary.downloaded}, Failed: {summary.failed}")
    if summary.failed > 0:
        raise typer.Exit(code=1)


@app.command(name="download_content_images")
def download_content_images(
    image_id: Optional[list[int]] = typer.Option(
        None,
        "--image-id",
        help="Content image IDs to download. Can be used multiple times.",
    ),
    all_images: bool = typer.Option(
        False,
        "--all",
        help="Download all content images listed in data/content-images.csv.",
    ),
    access_key: Optional[str] = typer.Option(
        None,
        "--access-key",
        envvar="UNSPLASH_ACCESS_KEY",
        help="Unsplash API access key. Can also be provided via UNSPLASH_ACCESS_KEY.",
    ),
) -> None:
    """Download content images using Unsplash API."""
    selected_ids = image_id or []
    if all_images and selected_ids:
        raise typer.BadParameter("Use either --image-id (multiple allowed) or --all, not both.")
    if not all_images and not selected_ids:
        raise typer.BadParameter("Provide at least one --image-id or use --all.")
    if not access_key:
        raise typer.BadParameter("Missing Unsplash access key. Set --access-key or UNSPLASH_ACCESS_KEY.")

    repo_root = _repo_root()
    csv_path = repo_root / "data" / "content-images.csv"
    out_dir = repo_root / "data" / "content-images"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_cols = {"image_id", "image_url"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise typer.BadParameter(f"Missing required column(s) in {csv_path.name}: {missing_list}")

    if all_images:
        target_df = df.copy()
    else:
        requested = set(selected_ids)
        target_df = df[df["image_id"].isin(requested)].copy()
        found = set(int(v) for v in target_df["image_id"].tolist())
        missing_ids = sorted(requested - found)
        if missing_ids:
            raise typer.BadParameter("Unknown image_id(s): " + ", ".join(str(v) for v in missing_ids))

    pad_width = max(2, len(str(int(df["image_id"].max()))))
    api_headers = {
        "Accept-Version": "v1",
        "User-Agent": "modelretrieval-1-subtask-b/1.0",
    }

    for row in target_df.itertuples(index=False):
        numeric_id = int(row.image_id)
        photo_page_url = str(row.image_url)
        padded_id = str(numeric_id).zfill(pad_width)
        destination = out_dir / f"{padded_id}.jpg"

        try:
            photo_id = _extract_unsplash_photo_id(photo_page_url)
            photo_endpoint = f"https://api.unsplash.com/photos/{photo_id}?" f"{urlencode({'client_id': access_key})}"
            photo_json = _get_json(photo_endpoint, headers=api_headers)

            download_location = photo_json.get("links", {}).get("download_location")
            if not download_location:
                raise ValueError(f"No download_location found for photo id {photo_id}")

            tracked_download_url = _append_client_id(download_location, access_key)
            tracked_download_json = _get_json(tracked_download_url, headers=api_headers)

            image_download_url = tracked_download_json.get("url")
            if not image_download_url:
                raise ValueError(f"No direct download URL returned for photo id {photo_id}")

            image_data = _download_bytes(image_download_url, headers=api_headers)
            destination.write_bytes(image_data)
            typer.echo(f"Downloaded image_id={numeric_id} -> {destination.relative_to(repo_root)}")
        except (HTTPError, URLError, ValueError, json.JSONDecodeError) as exc:
            typer.echo(
                f"Failed image_id={numeric_id} ({photo_page_url}): {exc}",
                err=True,
            )
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
