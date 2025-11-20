"""Utility to mine Reader Service images from Internet Archive and build a Hugging Face dataset."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests
from datasets import Dataset, DatasetDict, Features, Image, Value
from tqdm import tqdm
import pypdfium2 as pdfium

IA_METADATA_URL = "https://archive.org/metadata/{identifier}"
IA_DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"
IA_ADVSEARCH_URL = "https://archive.org/advancedsearch.php"
DOCUMENT_EXTENSIONS = (".pdf",)

LOGGER = logging.getLogger("readerservice_miner")


def sanitize_token(value: str, fallback: str) -> str:
    """Return a filesystem-safe token derived from ``value``."""

    sanitized = "".join(c for c in value if c.isalnum() or c in ("-", "_"))
    return sanitized or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Internet Archive Reader Service scans and materialize a Hugging Face dataset."
    )
    parser.add_argument(
        "--collection",
        default="readerservice",
        help="Internet Archive identifier to mine (default: readerservice).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where raw images, manifest, and the HF dataset will be written.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images to download from the collection.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1024,
        help="Skip files smaller than this many bytes (default: 1024).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for the generated dataset (0 disables validation split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed applied before splitting the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download images even if they already exist locally.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=1 << 16,
        help="Download chunk size in bytes (default: 65536).",
    )
    return parser.parse_args()


def fetch_collection_metadata(identifier: str) -> Dict:
    resp = requests.get(IA_METADATA_URL.format(identifier=identifier), timeout=60)
    resp.raise_for_status()
    return resp.json()


def iter_collection_items(
    collection: str, *, page_size: int = 500, progress: tqdm | None = None
) -> Iterable[str]:
    page = 1
    total_items = None
    while True:
        params = {
            "q": f"collection:{collection}",
            "fl[]": "identifier",
            "sort[]": "identifier asc",
            "rows": page_size,
            "page": page,
            "output": "json",
        }
        resp = requests.get(IA_ADVSEARCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        docs = payload.get("response", {}).get("docs", [])
        num_found = payload.get("response", {}).get("numFound", 0)
        if total_items is None:
            total_items = max(0, num_found - 1)
            if progress is not None:
                progress.reset(total=total_items or None)
        if not docs:
            break
        for doc in docs:
            identifier = doc.get("identifier")
            if identifier and identifier != collection:
                if progress is not None:
                    progress.update(1)
                yield identifier
        if page * page_size >= num_found:
            break
        page += 1


def gather_collection_files(
    collection: str,
    *,
    min_bytes: int,
    max_images: int | None,
) -> Tuple[List[Tuple[str, Dict]], int]:
    files: List[Tuple[str, Dict]] = []
    docs_with_images = set()
    with tqdm(desc="Discovering documents", unit="doc", total=0) as progress:
        for identifier in iter_collection_items(collection, progress=progress):
            try:
                metadata = fetch_collection_metadata(identifier)
            except requests.RequestException as exc:
                LOGGER.warning("Failed to fetch metadata for %s: %s", identifier, exc)
                continue
            image_files = list(iter_document_files(metadata, min_bytes=min_bytes))
            if not image_files:
                continue
            docs_with_images.add(identifier)
            progress.set_postfix(doc_with_images=len(docs_with_images), images=len(files))
            for file_info in image_files:
                files.append((identifier, file_info))
                if max_images and len(files) >= max_images:
                    return files, len(docs_with_images)
    return files, len(docs_with_images)


def iter_document_files(metadata: Dict, *, min_bytes: int) -> Iterable[Dict]:
    for file_info in metadata.get("files", []):
        name = file_info.get("name")
        if not name:
            continue
        lower = name.lower()
        fmt = (file_info.get("format") or "").lower()
        if not (lower.endswith(DOCUMENT_EXTENSIONS) or "pdf" in fmt):
            continue
        size = int(file_info.get("size") or 0)
        if size < min_bytes:
            continue
        yield {
            "name": name,
            "size": size,
            "format": fmt,
            "md5": file_info.get("md5") or "",
            "sha1": file_info.get("sha1") or "",
        }


def safe_local_path(root: Path, name: str, fallback_index: int) -> Path:
    """Return a path inside ``root`` that is safe to write to."""
    candidate = Path(name).name or f"reader_{fallback_index:05d}.jpg"
    stem = Path(candidate).stem or f"reader_{fallback_index:05d}"
    suffix = Path(candidate).suffix or ".jpg"
    sanitized = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))
    if not sanitized:
        sanitized = f"reader_{fallback_index:05d}"
    out_path = root / f"{sanitized}{suffix.lower()}"
    counter = 1
    while out_path.exists():
        out_path = root / f"{sanitized}_{counter:03d}{suffix.lower()}"
        counter += 1
    return out_path


def download_document(
    session: requests.Session,
    identifier: str,
    file_info: Dict,
    destination_dir: Path,
    *,
    overwrite: bool,
    chunk_bytes: int,
    index: int,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_path = safe_local_path(destination_dir, file_info["name"], index)
    if local_path.exists() and not overwrite:
        LOGGER.debug("Skipping existing %s", local_path)
        return local_path
    url = IA_DOWNLOAD_URL.format(identifier=identifier, filename=file_info["name"])
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as handle:
            for chunk in resp.iter_content(chunk_bytes):
                if chunk:
                    handle.write(chunk)
    return local_path


def convert_document_to_images(
    document_path: Path,
    *,
    image_root: Path,
    identifier: str,
    overwrite: bool,
    dpi: int,
) -> List[Path]:
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_to_images(
            document_path,
            image_root=image_root,
            identifier=identifier,
            overwrite=overwrite,
            dpi=dpi,
        )
    raise RuntimeError(f"Unsupported document type: {suffix or 'unknown'}")


def render_pdf_to_images(
    document_path: Path,
    *,
    image_root: Path,
    identifier: str,
    overwrite: bool,
    dpi: int,
) -> List[Path]:
    token = sanitize_token(identifier, "reader")
    prefix = sanitize_token(document_path.stem or token, token)
    doc_dir = image_root / token
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(document_path))
    page_paths: List[Path] = []
    scale = max(dpi, 72) / 72.0
    for page_number in range(len(pdf)):
        page = pdf[page_number]
        out_path = doc_dir / f"{prefix}_page_{page_number + 1:04d}.jpg"
        if out_path.exists() and not overwrite:
            page_paths.append(out_path)
            page.close()
            continue
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        pil_image.save(out_path, format="JPEG")
        bitmap.close()
        page.close()
        page_paths.append(out_path)
    pdf.close()
    return page_paths


def build_records(
    *,
    collection: str,
    files: Sequence[Tuple[str, Dict]],
    image_dest: Path,
    document_dest: Path,
    overwrite: bool,
    chunk_bytes: int,
    render_dpi: int = 200,
) -> List[Dict]:
    session = requests.Session()
    records: List[Dict] = []
    document_dest.mkdir(parents=True, exist_ok=True)
    image_dest.mkdir(parents=True, exist_ok=True)
    for idx, (identifier, file_info) in enumerate(tqdm(files, desc="Downloading documents")):
        try:
            local_path = download_document(
                session,
                identifier,
                file_info,
                document_dest,
                overwrite=overwrite,
                chunk_bytes=chunk_bytes,
                index=idx,
            )
        except requests.RequestException as exc:
            LOGGER.warning("Failed to download %s: %s", file_info.get("name"), exc)
            continue
        try:
            page_paths = convert_document_to_images(
                local_path,
                image_root=image_dest,
                identifier=identifier,
                overwrite=overwrite,
                dpi=render_dpi,
            )
        except RuntimeError as exc:
            LOGGER.warning("Failed to convert %s into images: %s", local_path, exc)
            continue
        if not page_paths:
            LOGGER.warning("No pages were rendered from %s; skipping", local_path)
            continue
        for page_idx, page_path in enumerate(page_paths):
            record = {
                "image": str(page_path),
                "page_index": page_idx,
                "filename": file_info["name"],
                "original_url": IA_DOWNLOAD_URL.format(
                    identifier=identifier, filename=file_info["name"]
                ),
                "bytes": page_path.stat().st_size,
                "md5": file_info.get("md5") or "",
                "sha1": file_info.get("sha1") or "",
                "source": collection,
                "document": identifier,
            }
            records.append(record)
    return records


def build_hf_dataset(records: Sequence[Dict], *, val_fraction: float, seed: int, output_dir: Path) -> Path:
    if not records:
        raise ValueError("No images were downloaded; cannot create dataset.")
    features = Features(
        {
            "image": Image(),
            "page_index": Value("int32"),
            "filename": Value("string"),
            "original_url": Value("string"),
            "bytes": Value("int64"),
            "md5": Value("string"),
            "sha1": Value("string"),
            "source": Value("string"),
            "document": Value("string"),
        }
    )
    columns: Dict[str, List] = {key: [] for key in features.keys()}
    for record in records:
        for key in columns:
            columns[key].append(record[key])
    dataset = Dataset.from_dict(columns, features=features)
    dataset = dataset.shuffle(seed=seed)
    ds_dict = DatasetDict()
    val_fraction = max(0.0, min(0.95, val_fraction))
    if val_fraction > 0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=val_fraction, seed=seed)
        ds_dict["train"] = split["train"]
        ds_dict["validation"] = split["test"]
    else:
        ds_dict["train"] = dataset
    ds_path = output_dir / "hf_dataset"
    ds_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(ds_path)
    return ds_path


def write_manifest(
    *,
    output_dir: Path,
    collection: str,
    metadata: Dict,
    records: Sequence[Dict],
    dataset_path: Path,
    val_fraction: float,
) -> None:
    manifest = {
        "collection": collection,
        "title": metadata.get("metadata", {}).get("title", ""),
        "source_url": f"https://archive.org/details/{collection}",
        "num_images": len(records),
        "hf_dataset_path": str(dataset_path),
        "val_fraction": val_fraction,
        "fields": list(records[0].keys()) if records else [],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = out_dir / "images"
    document_dir = out_dir / "documents"
    LOGGER.info("Fetching metadata for %s", args.collection)
    metadata = fetch_collection_metadata(args.collection)
    LOGGER.info("Discovering items within the collection")
    files, num_documents = gather_collection_files(
        args.collection,
        min_bytes=args.min_bytes,
        max_images=args.max_images,
    )
    if not files:
        raise SystemExit("No qualifying images found in the specified collection.")
    LOGGER.info(
        "Preparing to download %d images spanning %d documents",
        len(files),
        num_documents,
    )
    records = build_records(
        collection=args.collection,
        files=files,
        image_dest=image_dir,
        document_dest=document_dir,
        overwrite=args.overwrite,
        chunk_bytes=args.chunk_bytes,
    )
    if not records:
        raise SystemExit("No images were downloaded; aborting dataset creation.")
    LOGGER.info("Building Hugging Face dataset")
    ds_path = build_hf_dataset(
        records,
        val_fraction=args.val_fraction,
        seed=args.seed,
        output_dir=out_dir,
    )
    write_manifest(
        output_dir=out_dir,
        collection=args.collection,
        metadata=metadata,
        records=records,
        dataset_path=ds_path,
        val_fraction=args.val_fraction,
    )
    LOGGER.info("Done! Dataset saved to %s", ds_path)


if __name__ == "__main__":
    main()
