from __future__ import annotations
import json
import re
import threading
import time
import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, List, Union, TypeAlias
import errno
import time
from typing import Dict

JSONPath = Union[str, Path]  

class DescripitionBuilder:

    @staticmethod
    def json_description_builder(image_id: str, xml: str, assetsubtype: str, data_type: str) -> str:

        match data_type:
            case "Cyclomedia-Cubemap-Karo" | "Cyclomedia-Equirectangle-Yeni":
                dataset = xml["Dataset_Name"][0]
                create_time = xml["GeneratedAt"][0]

            case _:
                dataset = data_type
            
        payload: Dict[str, object] = {
            "type": "Objexa",
            "others": [
                f"ImageID: {image_id}, Dataset: {dataset}",
                {"assetsubtype": assetsubtype},
            ],
            "imageid": image_id,
        }

        if data_type == "Cyclomedia-Cubemap-Karo" or "Cyclomedia-Equirectangle-Yeni":
            payload["create_time"] = create_time

        return json.dumps(payload, ensure_ascii=False)

class _AsyncLastWriter:
    _lock       = threading.Lock()
    _buffer: dict[str, str] = {}     
    _json_path: Path | None = None
    _interval   = 2.0                
    _started    = False

    @classmethod
    def start(cls, json_path: JSONPath, *, interval: float = 2.0) -> None: 
        if cls._started:
            return
        cls._json_path = Path(json_path).expanduser().resolve()
        cls._interval  = float(interval)
        cls._started   = True

        def _worker() -> None:
            while True:
                time.sleep(cls._interval)
                cls.flush_now()

        threading.Thread(target=_worker, daemon=True).start()
        atexit.register(cls.flush_now)

    @classmethod
    def enqueue(cls, *, key: str, idx: int, total: int, filename: str) -> None:
        raw = f"({idx}/{total}): {filename}"
        with cls._lock:
            old_raw = cls._buffer.get(key)
            if old_raw:
                m = re.match(r"\((\d+)/", old_raw)
                old_idx = int(m.group(1)) if m else -1
                if idx <= old_idx:
                    return
            cls._buffer[key] = raw

    @classmethod
    def flush_now(cls) -> None:

        def _atomic_replace(tmp: Path, dst: Path, attempts: int = 5, backoff: float = 0.05):
            for i in range(attempts):
                try:
                    tmp.replace(dst)
                    return
                except PermissionError as e:
                    if i == attempts - 1 or e.winerror != 5:
                        raise
                    time.sleep(backoff * (2 ** i))
                if cls._json_path is None:
                    return
                
        with cls._lock:
            if not cls._buffer:
                return
            payload = cls._buffer.copy()
            cls._buffer.clear()

        p = cls._json_path
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            meta = {}

        meta.update(payload)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=4), encoding="utf-8")
        _atomic_replace(tmp, p)              

class LogUtils:
    DATE_FMT = "%Y-%m-%d %H:%M:%S"
    log_path: Path | None = None

    @staticmethod
    def init_async_writer(json_path: JSONPath, interval: float = 2.0) -> None:
        _AsyncLastWriter.start(json_path, interval=interval)

    @staticmethod
    def create(
        json_path: JSONPath,
        *,
        process_id: str,
        data_type: str,
        output_directory: JSONPath,
        model_weights: JSONPath,
        model_parameters: List[Any],
        start_date: Union[str, datetime],
        depth_preprocess: int = 0,
        image_preprocess: int = 0,
        detection_segmentation: int = 0,
        overwrite: bool = False,
    ) -> Path:
        p = Path(json_path).expanduser().resolve()
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} already exists; pass overwrite=True.")
        if isinstance(start_date, datetime):
            start_date = start_date.strftime(LogUtils.DATE_FMT)

        payload = {
            "Process ID": process_id,
            "Data Type": data_type,
            "Output Directory": str(Path(output_directory).expanduser().resolve()),
            "Model Weights": str(Path(model_weights).expanduser().resolve()),
            "Model Parameters": model_parameters,
            "Start Date": start_date,
            "Depth Preprocess": depth_preprocess,
            "Image Preprocess": image_preprocess,
            "Detection / Segmentation": detection_segmentation,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        return p

    @staticmethod
    def read(json_path: JSONPath) -> dict[str, Any]:
        try:
            return json.loads(Path(json_path).expanduser().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def update(json_path: JSONPath, **fields: Any) -> None:
        p = Path(json_path).expanduser().resolve()
        meta = LogUtils.read(p)
        meta.update(fields)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=4), encoding="utf-8")

    @staticmethod
    def delete(json_path: JSONPath) -> None:
        try:
            Path(json_path).expanduser().unlink()
        except FileNotFoundError:
            pass

    _progress_re = re.compile(r"\((\d+)/")

    @staticmethod
    def _parse_idx(raw: str) -> int:
        m = LogUtils._progress_re.search(raw)
        return int(m.group(1)) if m else -1

    @staticmethod
    def update_last_processed(
        json_path: JSONPath, *,
        key: str, idx: int, total: int, filename: str
    ) -> None:
        _AsyncLastWriter.enqueue(
            key=key, idx=idx, total=total, filename=filename
        )

    @staticmethod
    def resume_index(
        json_path: JSONPath,
        jpg_list: list[str],
        *,
        key: str,
    ) -> int:
        meta = LogUtils.read(json_path)
        raw = meta.get(key)
        if not raw:
            return 0
        m = re.match(r"\((\d+)/\d+\):\s*(.+)", raw)
        if not m:
            return 0
        last_name = m.group(2).strip()
        for i, path in enumerate(jpg_list):
            if Path(path).name == last_name:
                return i           
        return 0

class DecisionUtils:
    @staticmethod
    def decide_process_type(path: str) -> str:
        lower = path.lower()
        if "detection" in lower:
            return "detection"
        if "segmentation" in lower:
            return "segmentation"
        raise ValueError(
            "Path must contain either 'detection' or 'segmentation' "
            f"(got: {path!r})"
        )
