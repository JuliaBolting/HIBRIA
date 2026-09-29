"""Publicação atômica do par FAISS/metadados. Chamadores usam o mesmo lock."""
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def active_paths(directory):
    directory = Path(directory)
    pointer = directory / "CURRENT"
    if pointer.exists():
        name = pointer.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[a-f0-9]{32}", name):
            raise ValueError("Manifesto do FAISS inválido; execute o reparo.")
        directory = directory / "generations" / name
    return directory / "index.faiss", directory / "metadata.json"


def _fsync_dir(path):
    if os.name == "posix":
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish(directory, faiss, index, metadata):
    directory = Path(directory)
    if index.ntotal != len(metadata):
        raise ValueError("Índice e metadados divergentes; a gravação foi bloqueada.")
    previous = (directory / "CURRENT").read_text().strip() if (directory / "CURRENT").exists() else ""
    name = uuid.uuid4().hex
    target = directory / "generations" / name
    target.mkdir(parents=True)
    pointer_tmp = directory / ("CURRENT." + name + ".tmp")
    committed = False
    try:
        faiss.write_index(index, str(target / "index.faiss"))
        with (target / "index.faiss").open("rb") as file:
            os.fsync(file.fileno())
        with (target / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        _fsync_dir(target)
        _fsync_dir(target.parent)
        with pointer_tmp.open("w", encoding="ascii") as file:
            file.write(name + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(pointer_tmp, directory / "CURRENT")
        committed = True
        _fsync_dir(directory)
    finally:
        pointer_tmp.unlink(missing_ok=True)
        if not committed:
            shutil.rmtree(target, ignore_errors=True)
    # Conserva a geração anterior para recuperação, sem crescimento ilimitado.
    for old in target.parent.iterdir():
        if old.is_dir() and old.name not in {name, previous} and re.fullmatch(r"[a-f0-9]{32}", old.name):
            shutil.rmtree(old, ignore_errors=True)
    return target / "index.faiss", target / "metadata.json"
