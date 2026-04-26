from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

DEFAULT_OUTPUT_FILE = "inspect_yolo_pt_report.txt"


ARG_GROUPS = OrderedDict(
    [
        (
            "Base",
            [
                "task",
                "mode",
                "model",
                "data",
                "epochs",
                "time",
                "patience",
                "batch",
                "imgsz",
                "device",
                "workers",
                "project",
                "name",
                "exist_ok",
                "pretrained",
                "seed",
                "deterministic",
                "single_cls",
                "classes",
                "rect",
                "resume",
                "amp",
                "fraction",
                "freeze",
                "multi_scale",
                "overlap_mask",
                "mask_ratio",
                "dropout",
                "val",
                "split",
                "plots",
            ],
        ),
        (
            "Optimizacion",
            [
                "optimizer",
                "lr0",
                "lrf",
                "momentum",
                "weight_decay",
                "warmup_epochs",
                "warmup_momentum",
                "warmup_bias_lr",
                "nbs",
                "cos_lr",
                "close_mosaic",
            ],
        ),
        (
            "Loss",
            [
                "box",
                "cls",
                "dfl",
                "pose",
                "kobj",
                "label_smoothing",
                "cls_w",
                "o2m",
                "topk",
            ],
        ),
        (
            "Augmentations",
            [
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "flipud",
                "fliplr",
                "bgr",
                "mosaic",
                "mixup",
                "copy_paste",
                "copy_paste_mode",
                "auto_augment",
                "erasing",
                "crop_fraction",
                "augmentations",
            ],
        ),
        (
            "Guardado",
            [
                "save",
                "save_period",
                "cache",
                "verbose",
            ],
        ),
    ]
)

SNIPPET_ORDER = [
    "device",
    "data",
    "epochs",
    "time",
    "batch",
    "imgsz",
    "project",
    "name",
    "exist_ok",
    "pretrained",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "freeze",
    "patience",
    "deterministic",
    "single_cls",
    "classes",
    "rect",
    "cos_lr",
    "close_mosaic",
    "amp",
    "overlap_mask",
    "mask_ratio",
    "dropout",
    "box",
    "cls",
    "dfl",
    "pose",
    "kobj",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
    "crop_fraction",
]

SNIPPET_EXCLUDE = {
    "task",
    "mode",
    "model",
    "source",
    "save_dir",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show",
    "show_labels",
    "show_conf",
    "visualize",
    "embed",
    "stream_buffer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspecciona un checkpoint YOLO .pt y muestra hiperparametros, "
            "metadata y un snippet model.train(...) listo para copiar."
        )
    )
    parser.add_argument("model_path", help="Ruta al archivo .pt")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime toda la informacion en JSON en lugar de texto legible.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Incluye todos los train_args y las llaves completas del checkpoint.",
    )
    parser.add_argument(
        "--show-arch",
        action="store_true",
        help="Incluye el YAML completo de arquitectura si esta disponible.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Archivo donde se guardara el reporte. "
            f"Se sobreescribe en cada ejecucion. Default: {DEFAULT_OUTPUT_FILE}"
        ),
    )
    return parser.parse_args()


def make_json_safe(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return repr(value)

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"

    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v, depth + 1, max_depth) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item, depth + 1, max_depth) for item in value]

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return make_json_safe(value.item(), depth + 1, max_depth)
        except Exception:
            pass

    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return make_json_safe(value.tolist(), depth + 1, max_depth)
        except Exception:
            pass

    return repr(value)


def format_inline(value: Any) -> str:
    safe = make_json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, ensure_ascii=False)
    return str(safe)


def format_block(value: Any, indent: str = "  ") -> str:
    safe = make_json_safe(value)
    if isinstance(safe, (dict, list)):
        rendered = json.dumps(safe, ensure_ascii=False, indent=2)
    else:
        rendered = str(safe)
    return "\n".join(f"{indent}{line}" for line in rendered.splitlines())


def sha256_prefix(path: Path, max_bytes: int = 12) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[: max_bytes * 2]


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256_prefix": sha256_prefix(path),
    }


def normalize_names(names: Any) -> dict[Any, Any] | None:
    if isinstance(names, Mapping):
        return {k: v for k, v in names.items()}
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return None


def extract_runtime_model(ckpt: Mapping[str, Any]) -> Any:
    for key in ("ema", "model"):
        runtime_model = ckpt.get(key)
        if runtime_model is not None:
            return runtime_model
    return None


def extract_model_summary(runtime_model: Any, yolo_model: Any = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if yolo_model is not None:
        summary["task"] = getattr(yolo_model, "task", None)
        summary["ckpt_path"] = getattr(yolo_model, "ckpt_path", None)

    if runtime_model is None:
        return {k: v for k, v in summary.items() if v is not None}

    summary["module_type"] = type(runtime_model).__name__

    names = normalize_names(getattr(yolo_model, "names", None)) if yolo_model is not None else None
    if not names:
        names = normalize_names(getattr(runtime_model, "names", None))

    if names:
        summary["num_classes"] = len(names)
        summary["class_names"] = names
        summary["class_names_preview"] = dict(list(names.items())[:10])

    stride = getattr(runtime_model, "stride", None)
    if stride is not None:
        summary["stride"] = make_json_safe(stride)

    args = getattr(runtime_model, "args", None)
    if isinstance(args, Mapping):
        summary["effective_args_source"] = "model.args"

    yaml_data = getattr(runtime_model, "yaml", None)
    if isinstance(yaml_data, Mapping):
        summary["yaml_brief"] = {
            key: make_json_safe(yaml_data[key])
            for key in ("yaml_file", "scale", "depth_multiple", "width_multiple", "nc", "activation")
            if key in yaml_data
        }
        if "backbone" in yaml_data:
            summary["yaml_brief"]["backbone_blocks"] = len(yaml_data["backbone"])
        if "head" in yaml_data:
            summary["yaml_brief"]["head_blocks"] = len(yaml_data["head"])

    try:
        parameters = list(runtime_model.parameters())
        summary["total_params"] = sum(parameter.numel() for parameter in parameters)
        summary["trainable_params"] = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    except Exception:
        pass

    try:
        summary["module_count"] = sum(1 for _ in runtime_model.modules())
    except Exception:
        pass

    return {k: v for k, v in summary.items() if v is not None}


def group_train_args(train_args: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = OrderedDict()
    used_keys: set[str] = set()

    for group_name, keys in ARG_GROUPS.items():
        group_values = {key: train_args[key] for key in keys if key in train_args}
        if group_values:
            grouped[group_name] = group_values
            used_keys.update(group_values.keys())

    extras = {key: train_args[key] for key in sorted(train_args) if key not in used_keys}
    return grouped, extras


def build_train_snippet(model_path: str, train_args: Mapping[str, Any]) -> str | None:
    if not train_args:
        return None

    ordered_keys = [key for key in SNIPPET_ORDER if key in train_args and key not in SNIPPET_EXCLUDE]
    ordered_keys.extend(
        key
        for key in sorted(train_args)
        if key not in ordered_keys and key not in SNIPPET_EXCLUDE and not key.startswith("_")
    )

    snippet_lines = [f"model = YOLO({model_path!r})", "results = model.train("]
    added = 0

    for key in ordered_keys:
        value = train_args[key]
        safe_value = make_json_safe(value)
        if isinstance(safe_value, str) and safe_value.startswith("<") and safe_value.endswith(">"):
            continue
        snippet_lines.append(f"    {key}={pformat(safe_value, width=88)},")
        added += 1

    snippet_lines.append(")")
    return "\n".join(snippet_lines) if added else None


def build_payload(path: Path, ckpt: Mapping[str, Any], load_method: str, yolo_model: Any = None) -> dict[str, Any]:
    runtime_model = getattr(yolo_model, "model", None) if yolo_model is not None else extract_runtime_model(ckpt)
    train_args = ckpt.get("train_args")
    if not isinstance(train_args, Mapping):
        candidate_args = getattr(runtime_model, "args", None)
        train_args = candidate_args if isinstance(candidate_args, Mapping) else {}

    names = normalize_names(getattr(yolo_model, "names", None)) if yolo_model is not None else None
    if not names and runtime_model is not None:
        names = normalize_names(getattr(runtime_model, "names", None))

    model_summary = extract_model_summary(runtime_model=runtime_model, yolo_model=yolo_model)
    if names and "num_classes" not in model_summary:
        model_summary["num_classes"] = len(names)
        model_summary["class_names"] = names
        model_summary["class_names_preview"] = dict(list(names.items())[:10])

    grouped_args, extra_args = group_train_args(train_args)

    checkpoint_meta = {
        key: ckpt.get(key)
        for key in ("epoch", "best_fitness", "date", "version", "license", "docs")
        if ckpt.get(key) is not None
    }

    train_results = ckpt.get("train_results")
    payload = {
        "file": file_info(path),
        "load_method": load_method,
        "model_summary": model_summary,
        "checkpoint_meta": checkpoint_meta,
        "train_args": make_json_safe(dict(train_args)),
        "grouped_train_args": make_json_safe(grouped_args),
        "extra_train_args": make_json_safe(extra_args),
        "train_results": make_json_safe(train_results),
        "raw_checkpoint_keys": sorted(str(key) for key in ckpt.keys()),
        "train_snippet": build_train_snippet(str(path), train_args),
    }

    if runtime_model is not None:
        yaml_data = getattr(runtime_model, "yaml", None)
        if isinstance(yaml_data, Mapping):
            payload["model_yaml"] = make_json_safe(yaml_data)

    return payload


def load_with_ultralytics(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    yolo_model = YOLO(str(path))
    ckpt = getattr(yolo_model, "ckpt", None)
    if not isinstance(ckpt, Mapping):
        raise TypeError("Ultralytics no devolvio un checkpoint en formato dict.")
    return build_payload(path=path, ckpt=ckpt, load_method="ultralytics.YOLO", yolo_model=yolo_model)


def load_with_torch(path: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise TypeError("torch.load no devolvio un checkpoint en formato dict.")
    return build_payload(path=path, ckpt=ckpt, load_method="torch.load")


def inspect_model(path: Path) -> dict[str, Any]:
    errors: list[str] = []

    for loader in (load_with_ultralytics, load_with_torch):
        try:
            payload = loader(path)
            if errors:
                payload["warnings"] = errors
            return payload
        except Exception as exc:
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No se pudo cargar el checkpoint.\n"
        + "\n".join(f"- {message}" for message in errors)
        + "\nActiva tu venv con torch + ultralytics antes de volver a intentarlo."
    )


def append_section(lines: list[str], title: str) -> None:
    if lines:
        lines.append("")
    lines.append(title)
    lines.append("-" * len(title))


def append_mapping(lines: list[str], title: str, values: Mapping[str, Any]) -> None:
    if not values:
        return
    append_section(lines, title)
    for key, value in values.items():
        if isinstance(make_json_safe(value), (dict, list)):
            lines.append(f"{key}:")
            lines.extend(format_block(value).splitlines())
        else:
            lines.append(f"{key}: {format_inline(value)}")


def build_text_report(payload: Mapping[str, Any], show_full: bool, show_arch: bool) -> str:
    lines: list[str] = []

    append_mapping(lines, "Archivo", payload["file"])
    append_mapping(lines, "Carga", {"load_method": payload.get("load_method")})
    append_mapping(lines, "Resumen del modelo", payload.get("model_summary", {}))
    append_mapping(lines, "Metadata del checkpoint", payload.get("checkpoint_meta", {}))

    grouped_args = payload.get("grouped_train_args", {})
    if grouped_args:
        for group_name, group_values in grouped_args.items():
            append_mapping(lines, f"Train args - {group_name}", group_values)

    extras = payload.get("extra_train_args", {})
    if extras and show_full:
        append_mapping(lines, "Train args - Extras", extras)

    train_results = payload.get("train_results")
    if train_results:
        append_section(lines, "Resultados guardados")
        lines.extend(format_block(train_results).splitlines())

    if show_arch and payload.get("model_yaml"):
        append_section(lines, "YAML de arquitectura")
        lines.extend(format_block(payload["model_yaml"]).splitlines())

    append_section(lines, "Snippet para train_seg.py")
    snippet = payload.get("train_snippet")
    if snippet:
        lines.extend(snippet.splitlines())
    else:
        lines.append("No se pudo construir el snippet porque el checkpoint no trae train_args reutilizables.")

    if show_full:
        append_mapping(lines, "Llaves del checkpoint", {"keys": payload.get("raw_checkpoint_keys", [])})

    warnings = payload.get("warnings")
    if warnings:
        append_section(lines, "Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def render_output(payload: Mapping[str, Any], use_json: bool, show_full: bool, show_arch: bool) -> str:
    if use_json:
        return json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2) + "\n"
    return build_text_report(payload, show_full=show_full, show_arch=show_arch)


def write_output_file(output_path: Path, content: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path.resolve()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_path = Path(args.output)

    if not model_path.exists() or not model_path.is_file():
        raise SystemExit(f"No existe el archivo: {model_path}")

    if model_path.suffix.lower() != ".pt":
        raise SystemExit(f"Se esperaba un archivo .pt y llego: {model_path.name}")

    payload = inspect_model(model_path)
    rendered_output = render_output(payload, use_json=args.json, show_full=args.full, show_arch=args.show_arch)
    saved_path = write_output_file(output_path, rendered_output)
    print(rendered_output, end="")
    print(f"Reporte guardado en: {saved_path}")


if __name__ == "__main__":
    main()
