import tempfile
import unittest
from pathlib import Path
from engine.vault import atomic_json, seal, verify

class VaultIntegrity(unittest.TestCase):
    def test_volatile_ui_log_does_not_break_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'manifest.json', {'schema':1})
            (root/'gui_worker.log').write_text('working')
            (root/'stop_requested').write_text('stop')
            self.assertEqual(seal(root),1)
            (root/'gui_worker.log').write_text('finished')
            (root/'stop_requested').unlink()
            result=verify(root)
            self.assertEqual(result, {'manifest.json':True})
