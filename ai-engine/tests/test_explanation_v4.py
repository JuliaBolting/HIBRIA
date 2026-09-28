"""A política v4 de números obrigatórios foi substituída pelos testes v6.

Este teste impede a reintrodução de um prompt separado no laboratório antigo.
"""
from pathlib import Path
import unittest

class LegacyLabCompatibilityTests(unittest.TestCase):
    def test_old_lab_delegates_to_current_lab(self):
        source = (Path(__file__).resolve().parents[1] / "scripts/explanation_v4_lab.py").read_text()
        self.assertIn("from explanation_v6_lab import main", source)
        self.assertNotIn("requests.post", source)

if __name__ == "__main__":
    unittest.main()
