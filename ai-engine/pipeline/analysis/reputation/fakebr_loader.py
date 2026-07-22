# =============================================================================
# Descoberta pontual de fontes no Fake.Br Corpus.
#
# O corpus é usado somente para obter URLs das fontes. A nota não vem do Fake.Br:
# cada domínio é enviado individualmente ao mesmo SourceReputationService usado
# pelo pipeline normal.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .identity import normalize_domain


@dataclass
class FakeBrSource:
    domain: str
    sample_url: str
    occurrences: int = 0
    metadata_files: list[str] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "sample_url": self.sample_url,
            "occurrences": self.occurrences,
            "labels": sorted(self.labels),
            "metadata_files": self.metadata_files,
        }


class FakeBrSourceDiscovery:
    """Lê somente a segunda linha dos arquivos *-meta.txt, que contém a URL original."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)

    def discover(self, *, include_fake: bool = False) -> list[FakeBrSource]:
        folders = [("true", self.dataset_path / "full_texts" / "true-meta-information")]
        if include_fake:
            folders.append(("fake", self.dataset_path / "full_texts" / "fake-meta-information"))

        found: dict[str, FakeBrSource] = {}
        for label, folder in folders:
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*-meta.txt")):
                url = self._read_source_url(path)
                domain = self._domain(url)
                if not domain:
                    continue
                item = found.setdefault(domain, FakeBrSource(domain=domain, sample_url=url))
                item.occurrences += 1
                item.labels.add(label)
                if len(item.metadata_files) < 5:
                    item.metadata_files.append(str(path))

        return sorted(found.values(), key=lambda item: (-item.occurrences, item.domain))

    def diagnostics(self, *, include_fake: bool = False) -> dict:
        true_folder = self.dataset_path / "full_texts" / "true-meta-information"
        fake_folder = self.dataset_path / "full_texts" / "fake-meta-information"
        sources = self.discover(include_fake=include_fake)
        return {
            "dataset_path": str(self.dataset_path),
            "dataset_exists": self.dataset_path.exists(),
            "true_metadata_folder": str(true_folder),
            "true_metadata_files": len(list(true_folder.glob("*-meta.txt"))) if true_folder.exists() else 0,
            "fake_metadata_folder": str(fake_folder),
            "fake_metadata_files": len(list(fake_folder.glob("*-meta.txt"))) if fake_folder.exists() else 0,
            "include_fake": include_fake,
            "unique_sources": len(sources),
            "source_preview": [item.to_dict() for item in sources[:20]],
        }

    @staticmethod
    def _read_source_url(path: Path) -> str:
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        except OSError:
            return ""
        if len(lines) < 2:
            return ""
        return lines[1].strip()

    @staticmethod
    def _domain(url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url if "://" in url else f"https://{url}")
        return normalize_domain(parsed.netloc or parsed.path)
