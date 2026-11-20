"""V7 Arena contract: only real episodes, each carrying its origin."""
import tempfile
import unittest
from pathlib import Path

from boolean_engine import native_backend as nb
from boolean_engine.rng import Rng
from engine.evolution import run
from engine.vault import read_json
from test_vector_engine import config


class RecordingsAreRealTraining(unittest.TestCase):
    def test_every_agent_is_recorded_on_the_same_training_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = run(tmp, config(generations=2))
            files = sorted((folder/'recordings').glob('*.json'))
            population = read_json(folder/'config.json')['population']
            self.assertEqual(len(files), 2 * (population + 1))
            for generation in (1, 2):
                items = [read_json(f) for f in files if f.name.startswith(f'g{generation:04d}_')]
                self.assertEqual(len({(i['scenario'], i['seed']) for i in items}), 1)
                for item in items:
                    # Origin of every frame: run kind, generation, actor, scenario, seed.
                    self.assertEqual(item['kind'], 'training')
                    self.assertEqual(item['generation'], generation)
                    self.assertTrue(item['actor'])
                    self.assertTrue(item['frames'])
                    self.assertEqual(item['frames'][0]['t'], 0)
                    self.assertTrue(all('action' in f for f in item['frames'][1:]))
            # Test episodes stay separate and are never listed as training.
            self.assertTrue(all(read_json(p).get('kind') != 'training' for p in (folder/'replays').glob('*.json')))

    def test_gui_has_no_simulation_path(self):
        source = Path(__file__).resolve().parents[1].joinpath('gui', 'laboratory.py').read_text(encoding='utf-8')
        for removed in ('preview_actors', 'load_preview', 'preview_action', 'Simulazione agenti'):
            self.assertNotIn(removed, source)


@unittest.skipUnless(nb.AVAILABLE, 'native kernel not built')
class SharedGenerator(unittest.TestCase):
    def test_python_and_rust_draw_the_same_numbers(self):
        import predatore_native
        for seed in (0, 1, 42, (1 << 64) - 5):
            r = Rng(seed)
            self.assertEqual(predatore_native.rng_draws(seed, 64, 0), [r.next_u64() for _ in range(64)])
            r = Rng(seed)
            self.assertEqual(predatore_native.rng_draws(seed, 64, 7), [r.below(7) for _ in range(64)])


if __name__ == '__main__':
    unittest.main()
