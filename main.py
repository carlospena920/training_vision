import re
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

# User configuration
dataset_folder = "Best_Seg_DefSide021326"  # Relative folder inside datasets/
ok_folder = "side_ok"  # Relative folder inside dataset_folder
model = "yolo26s --seg.pt"  # Base model for YOLO(...)
val_percentage = 20  # Reserved for upcoming steps
nok_percentage = 35  # Reserved for upcoming steps
dry_run = True  # True = simulate deletes, False = delete orphan files


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTENSIONS = {".txt"}


def build_file_map(directory: Path, allowed_extensions: Iterable[str]) -> DefaultDict[str, List[Path]]:
    file_map: DefaultDict[str, List[Path]] = defaultdict(list)
    allowed = {ext.lower() for ext in allowed_extensions}

    for item in directory.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() in allowed:
            file_map[item.stem.lower()].append(item)

    return file_map


def list_images_recursive(directory: Path) -> List[Path]:
    files = []
    for item in directory.rglob("*"):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(item)
    return sorted(files, key=lambda p: str(p).lower())


def validate_percentage(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > 100:
        raise ValueError(f"{name} must be between 0 and 100")


def validate_dataset_folder(dataset_name: str) -> Path:
    if not dataset_name or not dataset_name.strip():
        raise ValueError("dataset_folder cannot be empty")

    dataset_path = Path(dataset_name)
    if dataset_path.is_absolute():
        raise ValueError("dataset_folder must be relative to datasets/")
    if ".." in dataset_path.parts:
        raise ValueError("dataset_folder cannot contain '..'")

    return Path("datasets") / dataset_path


def step_1_sync_train_pairs(dataset_root: Path, dry_run_mode: bool) -> dict:
    print("\nPaso 1: Validar y sincronizar images/train vs labels/train")
    print(f"Dataset path evaluado: {dataset_root.resolve()}")

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"

    if not images_train.exists() or not images_train.is_dir():
        raise FileNotFoundError(f"Missing folder: {images_train}")
    if not labels_train.exists() or not labels_train.is_dir():
        raise FileNotFoundError(f"Missing folder: {labels_train}")

    image_map = build_file_map(images_train, IMAGE_EXTENSIONS)
    label_map = build_file_map(labels_train, LABEL_EXTENSIONS)

    image_stems = set(image_map.keys())
    label_stems = set(label_map.keys())

    paired_stems = image_stems & label_stems
    image_orphans = sorted(image_stems - label_stems)
    label_orphans = sorted(label_stems - image_stems)

    image_orphan_files = sorted(
        (path for stem in image_orphans for path in image_map[stem]),
        key=lambda p: str(p).lower(),
    )
    label_orphan_files = sorted(
        (path for stem in label_orphans for path in label_map[stem]),
        key=lambda p: str(p).lower(),
    )
    orphan_files = image_orphan_files + label_orphan_files

    print(f"Imagenes detectadas: {sum(len(paths) for paths in image_map.values())}")
    print(f"Labels detectados: {sum(len(paths) for paths in label_map.values())}")
    print(f"Huerfanos imagen: {len(image_orphan_files)}")
    print(f"Huerfanos label: {len(label_orphan_files)}")

    action = "se borrarian" if dry_run_mode else "se borraran"
    if orphan_files:
        print(f"\nArchivos que {action}:")
        for orphan in orphan_files:
            print(f"- {orphan}")
    else:
        print("\nNo hay archivos huerfanos para limpiar.")

    deleted_count = 0
    delete_errors = 0

    if not dry_run_mode:
        for orphan in orphan_files:
            try:
                orphan.unlink()
                deleted_count += 1
            except OSError as exc:
                delete_errors += 1
                print(f"ERROR deleting {orphan}: {exc}")

    valid_images = sum(len(image_map[stem]) for stem in paired_stems)
    valid_labels = sum(len(label_map[stem]) for stem in paired_stems)

    print("\nResumen final")
    print(f"Total imagenes validas: {valid_images}")
    print(f"Total labels validos: {valid_labels}")
    print(f"Huerfanos imagen: {len(image_orphan_files)}")
    print(f"Huerfanos label: {len(label_orphan_files)}")
    if dry_run_mode:
        print(f"Archivos que se borrarian: {len(orphan_files)}")
    else:
        print(f"Archivos borrados: {deleted_count}")
    print(f"Errores de borrado: {delete_errors}")

    return {
        "valid_images": valid_images,
        "valid_labels": valid_labels,
        "image_orphans": len(image_orphan_files),
        "label_orphans": len(label_orphan_files),
        "would_delete": len(orphan_files) if dry_run_mode else 0,
        "deleted": deleted_count,
        "delete_errors": delete_errors,
    }


def _upsert_top_level_yaml_value(lines: List[str], key: str, value: str) -> bool:
    key_prefix = f"{key}:"
    replacement = f"{key}: {value}\n"

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(key_prefix):
            continue

        leading_spaces = len(line) - len(stripped)
        if leading_spaces == 0:
            lines[index] = replacement
            return True

    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    lines.append(replacement)
    return False


def _remove_blank_line_before_key(lines: List[str], key: str) -> None:
    key_prefix = f"{key}:"
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(key_prefix) and (len(line) - len(stripped) == 0):
            if index > 0 and not lines[index - 1].strip():
                del lines[index - 1]
            return


def _single_file_per_stem(file_map: DefaultDict[str, List[Path]], kind: str) -> dict:
    duplicates = [stem for stem, paths in file_map.items() if len(paths) != 1]
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise ValueError(
            f"Expected 1 file per stem in {kind}, found duplicates for {len(duplicates)} stem(s): {sample}"
        )
    return {stem: paths[0] for stem, paths in file_map.items()}


def _print_progress(prefix: str, current: int, total: int, bar_width: int = 30) -> None:
    if total <= 0:
        print(f"{prefix}: 0/0")
        return

    ratio = current / float(total)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{prefix}: [{bar}] {current}/{total} ({ratio * 100:6.2f}%)", end=end, flush=True)


def _to_python_double_quoted(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _load_split_pair_maps(images_dir: Path, labels_dir: Path, split_name: str) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    image_map = _single_file_per_stem(build_file_map(images_dir, IMAGE_EXTENSIONS), f"images/{split_name}")
    label_map = _single_file_per_stem(build_file_map(labels_dir, LABEL_EXTENSIONS), f"labels/{split_name}")

    image_stems = set(image_map.keys())
    label_stems = set(label_map.keys())
    if image_stems != label_stems:
        image_orphans = sorted(image_stems - label_stems)
        label_orphans = sorted(label_stems - image_stems)
        raise ValueError(
            f"{split_name} folders are not synchronized. "
            f"image_orphans={len(image_orphans)}, label_orphans={len(label_orphans)}."
        )

    return image_map, label_map


def _classify_stems_from_label_content(label_map: Dict[str, Path]) -> Tuple[Set[str], Set[str]]:
    ok_stems: Set[str] = set()
    nok_stems: Set[str] = set()

    for stem, label_path in label_map.items():
        content = label_path.read_text(encoding="utf-8").strip()
        if content:
            nok_stems.add(stem)
        else:
            ok_stems.add(stem)

    return ok_stems, nok_stems


def _target_ok_count_from_nok(nok_count: int, nok_percentage_value: int) -> int:
    if nok_percentage_value == 0:
        if nok_count > 0:
            raise ValueError("nok_percentage=0 is invalid when there are NOK images.")
        return 0

    total_target = math.floor((nok_count * 100.0) / nok_percentage_value)
    return total_target - nok_count


def _move_paired_stems(
    stems_to_move: List[str], source_image_map: dict, source_label_map: dict, target_images_dir: Path, target_labels_dir: Path
) -> int:
    operations = []
    for stem in stems_to_move:
        image_src = source_image_map[stem]
        label_src = source_label_map[stem]
        image_dst = target_images_dir / image_src.name
        label_dst = target_labels_dir / label_src.name

        if image_dst.exists():
            raise FileExistsError(f"Target image already exists: {image_dst}")
        if label_dst.exists():
            raise FileExistsError(f"Target label already exists: {label_dst}")

        operations.append((image_src, image_dst, label_src, label_dst))

    moved_records = []
    total_ops = len(operations)
    if total_ops > 0:
        _print_progress("Progreso movimiento pares", 0, total_ops)
    try:
        for index, (image_src, image_dst, label_src, label_dst) in enumerate(operations, start=1):
            image_src.rename(image_dst)
            try:
                label_src.rename(label_dst)
            except Exception:
                image_dst.rename(image_src)
                raise
            moved_records.append((image_src, image_dst, label_src, label_dst))
            _print_progress("Progreso movimiento pares", index, total_ops)
    except Exception:
        for image_src, image_dst, label_src, label_dst in reversed(moved_records):
            if image_dst.exists():
                image_dst.rename(image_src)
            if label_dst.exists():
                label_dst.rename(label_src)
        raise

    return len(operations)


def step_3_split_nok_train_val(dataset_root: Path, val_percentage_value: int) -> None:
    print("\nPaso 3: Dividir NOK entre train y val")
    validate_percentage("val_percentage", val_percentage_value)

    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    images_val = dataset_root / "images" / "val"
    labels_val = dataset_root / "labels" / "val"

    for folder in (images_train, labels_train, images_val, labels_val):
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

    train_image_map, train_label_map = _load_split_pair_maps(images_train, labels_train, "train")
    val_image_map, val_label_map = _load_split_pair_maps(images_val, labels_val, "val")

    train_ok_stems, train_nok_stems = _classify_stems_from_label_content(train_label_map)
    val_ok_stems, val_nok_stems = _classify_stems_from_label_content(val_label_map)

    overlap_stems = (set(train_image_map.keys()) & set(val_image_map.keys()))
    if overlap_stems:
        sample = ", ".join(sorted(overlap_stems)[:5])
        raise ValueError(f"Found duplicated stems in train and val: {sample}")

    total_nok = len(train_nok_stems) + len(val_nok_stems)
    target_val_nok = math.floor((total_nok * val_percentage_value) / 100.0)
    current_val_nok = len(val_nok_stems)
    delta = target_val_nok - current_val_nok

    print(f"NOK train actuales: {len(train_nok_stems)}")
    print(f"NOK val actuales: {current_val_nok}")
    print(f"NOK totales: {total_nok}")
    print(f"Target NOK en val (redondeado hacia abajo): {target_val_nok}")

    if delta == 0:
        print("NOK ya estan divididos segun val_percentage. No se moveran NOK.")
        return

    if delta > 0:
        if delta > len(train_nok_stems):
            raise ValueError(f"Cannot move {delta} NOK pair(s) from train to val; only {len(train_nok_stems)}.")
        selected_stems = sorted(train_nok_stems)[:delta]
        moved = _move_paired_stems(selected_stems, train_image_map, train_label_map, images_val, labels_val)
        print(f"NOK movidos de train -> val: {moved}")
        return

    move_back = -delta
    if move_back > len(val_nok_stems):
        raise ValueError(f"Cannot move {move_back} NOK pair(s) from val to train; only {len(val_nok_stems)}.")
    selected_stems = sorted(val_nok_stems)[:move_back]
    moved = _move_paired_stems(selected_stems, val_image_map, val_label_map, images_train, labels_train)
    print(f"NOK movidos de val -> train: {moved}")


def step_2_update_data_yaml(dataset_root: Path) -> None:
    print("\nPaso 2: Actualizar data.yaml (path/train/val)")

    data_yaml_path = dataset_root / "data.yaml"
    if not data_yaml_path.exists() or not data_yaml_path.is_file():
        raise FileNotFoundError(f"Missing file: {data_yaml_path}")

    lines = data_yaml_path.read_text(encoding="utf-8").splitlines(keepends=True)

    dataset_path_value = dataset_root.as_posix()
    replaced_path = _upsert_top_level_yaml_value(lines, "path", dataset_path_value)
    replaced_train = _upsert_top_level_yaml_value(lines, "train", "images/train")
    replaced_val = _upsert_top_level_yaml_value(lines, "val", "images/val")
    _remove_blank_line_before_key(lines, "val")

    data_yaml_path.write_text("".join(lines), encoding="utf-8")

    print(f"Archivo actualizado: {data_yaml_path}")
    print(f"path: {dataset_path_value} ({'reemplazado' if replaced_path else 'agregado'})")
    print(f"train: images/train ({'reemplazado' if replaced_train else 'agregado'})")
    print(f"val: images/val ({'reemplazado' if replaced_val else 'agregado'})")


def _validate_step_4_ratios_and_pairs(dataset_root: Path, nok_percentage_value: int) -> None:
    images_train = dataset_root / "images" / "train"
    labels_train = dataset_root / "labels" / "train"
    images_val = dataset_root / "images" / "val"
    labels_val = dataset_root / "labels" / "val"

    train_image_map, train_label_map = _load_split_pair_maps(images_train, labels_train, "train")
    val_image_map, val_label_map = _load_split_pair_maps(images_val, labels_val, "val")

    train_ok_stems, train_nok_stems = _classify_stems_from_label_content(train_label_map)
    val_ok_stems, val_nok_stems = _classify_stems_from_label_content(val_label_map)

    expected_train_ok = _target_ok_count_from_nok(len(train_nok_stems), nok_percentage_value)
    expected_val_ok = _target_ok_count_from_nok(len(val_nok_stems), nok_percentage_value)

    if len(train_ok_stems) != expected_train_ok:
        raise ValueError(
            "Train ratio validation failed. "
            f"expected_ok={expected_train_ok}, current_ok={len(train_ok_stems)}, nok={len(train_nok_stems)}."
        )
    if len(val_ok_stems) != expected_val_ok:
        raise ValueError(
            "Val ratio validation failed. "
            f"expected_ok={expected_val_ok}, current_ok={len(val_ok_stems)}, nok={len(val_nok_stems)}."
        )

    train_total = len(train_image_map)
    val_total = len(val_image_map)
    train_ok_pct = (len(train_ok_stems) * 100.0 / train_total) if train_total else 0.0
    train_nok_pct = (len(train_nok_stems) * 100.0 / train_total) if train_total else 0.0
    val_ok_pct = (len(val_ok_stems) * 100.0 / val_total) if val_total else 0.0
    val_nok_pct = (len(val_nok_stems) * 100.0 / val_total) if val_total else 0.0

    print("Validacion final Paso 4:")
    print(
        f"train -> total={train_total}, ok={len(train_ok_stems)} ({train_ok_pct:.2f}%), "
        f"nok={len(train_nok_stems)} ({train_nok_pct:.2f}%)"
    )
    print(
        f"val -> total={val_total}, ok={len(val_ok_stems)} ({val_ok_pct:.2f}%), "
        f"nok={len(val_nok_stems)} ({val_nok_pct:.2f}%)"
    )
    print("Validacion de contrapartes images/labels en val: OK")


def step_4_add_ok_images_for_nok_percentage(
    dataset_root: Path, ok_folder_name: str, nok_percentage_value: int
) -> None:
    print("\nPaso 4: Completar imagenes OK por split segun nok_percentage")
    validate_percentage("nok_percentage", nok_percentage_value)

    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"
    ok_source = dataset_root / ok_folder_name

    for folder in (images_train, images_val, labels_train, labels_val):
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
    if not ok_source.exists() or not ok_source.is_dir():
        raise FileNotFoundError(f"Missing ok_folder: {ok_source}")

    train_image_map, train_label_map = _load_split_pair_maps(images_train, labels_train, "train")
    val_image_map, val_label_map = _load_split_pair_maps(images_val, labels_val, "val")

    train_ok_stems, train_nok_stems = _classify_stems_from_label_content(train_label_map)
    val_ok_stems, val_nok_stems = _classify_stems_from_label_content(val_label_map)

    target_train_ok = _target_ok_count_from_nok(len(train_nok_stems), nok_percentage_value)
    target_val_ok = _target_ok_count_from_nok(len(val_nok_stems), nok_percentage_value)

    need_train_ok = target_train_ok - len(train_ok_stems)
    need_val_ok = target_val_ok - len(val_ok_stems)

    print(
        f"train -> nok={len(train_nok_stems)}, ok_actual={len(train_ok_stems)}, "
        f"ok_objetivo={target_train_ok}, ok_faltantes={need_train_ok}"
    )
    print(
        f"val -> nok={len(val_nok_stems)}, ok_actual={len(val_ok_stems)}, "
        f"ok_objetivo={target_val_ok}, ok_faltantes={need_val_ok}"
    )

    if need_train_ok < 0:
        raise ValueError(
            f"Train has more OK than target. current_ok={len(train_ok_stems)}, target_ok={target_train_ok}."
        )
    if need_val_ok < 0:
        raise ValueError(f"Val has more OK than target. current_ok={len(val_ok_stems)}, target_ok={target_val_ok}.")

    total_needed = need_train_ok + need_val_ok
    if total_needed == 0:
        print("No se requieren imagenes OK adicionales.")
        _validate_step_4_ratios_and_pairs(dataset_root, nok_percentage_value)
        return

    candidate_images = list_images_recursive(ok_source)
    candidate_by_stem: DefaultDict[str, List[Path]] = defaultdict(list)
    for path in candidate_images:
        candidate_by_stem[path.stem.lower()].append(path)

    duplicate_candidate_stems = [stem for stem, paths in candidate_by_stem.items() if len(paths) > 1]
    if duplicate_candidate_stems:
        sample = ", ".join(sorted(duplicate_candidate_stems)[:5])
        raise ValueError(f"ok_folder has duplicate stems: {sample}")

    existing_stems = set(train_image_map.keys()) | set(val_image_map.keys())
    available_candidates = []
    for stem, paths in candidate_by_stem.items():
        if stem in existing_stems:
            continue
        available_candidates.append(paths[0])
    available_candidates = sorted(available_candidates, key=lambda p: str(p).lower())

    if len(available_candidates) < total_needed:
        raise ValueError(
            f"Not enough OK images in {ok_source}. Needed={total_needed}, available={len(available_candidates)}."
        )

    selected_for_train = available_candidates[:need_train_ok]
    selected_for_val = available_candidates[need_train_ok : need_train_ok + need_val_ok]

    operations = []
    for src in selected_for_train:
        image_dst = images_train / src.name
        label_dst = labels_train / f"{src.stem}.txt"
        if image_dst.exists() or label_dst.exists():
            raise FileExistsError(f"Target already exists for stem {src.stem} in train.")
        operations.append((src, image_dst, label_dst))
    for src in selected_for_val:
        image_dst = images_val / src.name
        label_dst = labels_val / f"{src.stem}.txt"
        if image_dst.exists() or label_dst.exists():
            raise FileExistsError(f"Target already exists for stem {src.stem} in val.")
        operations.append((src, image_dst, label_dst))

    moved_records = []
    created_labels = []
    total_ops = len(operations)
    if total_ops > 0:
        _print_progress("Progreso movimiento OK", 0, total_ops)
    try:
        for index, (src, image_dst, label_dst) in enumerate(operations, start=1):
            src.rename(image_dst)
            moved_records.append((src, image_dst))
            label_dst.write_text("", encoding="utf-8")
            created_labels.append(label_dst)
            _print_progress("Progreso movimiento OK", index, total_ops)
    except Exception:
        for label_path in reversed(created_labels):
            if label_path.exists():
                label_path.unlink()
        for original_src, moved_dst in reversed(moved_records):
            if moved_dst.exists():
                moved_dst.rename(original_src)
        raise

    print(f"Imagenes OK agregadas a train: {len(selected_for_train)}")
    print(f"Imagenes OK agregadas a val: {len(selected_for_val)}")
    _validate_step_4_ratios_and_pairs(dataset_root, nok_percentage_value)


def step_5_update_train_seg(dataset_folder_name: str, model_name: str, train_seg_path: Path = Path("train_seg.py")) -> None:
    print("\nPaso 5: Actualizar train_seg.py (run_name/model)")

    if not train_seg_path.exists() or not train_seg_path.is_file():
        raise FileNotFoundError(f"Missing file: {train_seg_path}")

    content = train_seg_path.read_text(encoding="utf-8")

    run_name_pattern = re.compile(r"(?m)^(?P<indent>\s*)run_name\s*=\s*.*$")
    yolo_pattern = re.compile(r"YOLO\(\s*(['\"]).*?\1\s*\)")

    run_name_line = f'run_name = {_to_python_double_quoted(dataset_folder_name)}'
    updated_content, run_replacements = run_name_pattern.subn(
        lambda m: f"{m.group('indent')}{run_name_line}", content, count=1
    )
    if run_replacements == 0:
        raise ValueError("Could not find run_name assignment in train_seg.py")

    yolo_call = f"YOLO({_to_python_double_quoted(model_name)})"
    updated_content, yolo_replacements = yolo_pattern.subn(yolo_call, updated_content, count=1)
    if yolo_replacements == 0:
        raise ValueError("Could not find YOLO('...') call in train_seg.py")

    if updated_content == content:
        print("train_seg.py ya estaba actualizado.")
        return

    train_seg_path.write_text(updated_content, encoding="utf-8")
    print(f"Archivo actualizado: {train_seg_path}")
    print(f"run_name -> {dataset_folder_name}")
    print(f"YOLO(...) -> {model_name}")


def main() -> None:
    print("Flujo de automatizacion de dataset")
    print(f"dataset_folder={dataset_folder}")
    print(f"ok_folder={ok_folder}")
    print(f"model={model}")
    print(f"val_percentage={val_percentage}")
    print(f"nok_percentage={nok_percentage}")
    print(f"dry_run={dry_run}")

    dataset_root = validate_dataset_folder(dataset_folder)
    step_1_sync_train_pairs(dataset_root=dataset_root, dry_run_mode=dry_run)
    step_2_update_data_yaml(dataset_root=dataset_root)
    step_3_split_nok_train_val(dataset_root=dataset_root, val_percentage_value=val_percentage)
    step_4_add_ok_images_for_nok_percentage(
        dataset_root=dataset_root, ok_folder_name=ok_folder, nok_percentage_value=nok_percentage
    )
    step_5_update_train_seg(dataset_folder_name=dataset_folder, model_name=model)


if __name__ == "__main__":
    main()
