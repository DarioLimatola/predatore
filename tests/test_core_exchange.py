import json
import tempfile
import unittest
from pathlib import Path

from boolean_engine import VectorBooleanNet
from boolean_engine.core_adapter import StructuralCounter, encode_tensor, decode_tensor
from engine.candidate_exchange import (digest, circuit_candidate, experience_candidate, import_candidate,
                                       evaluate_candidate)
from tools.candidate_exchange import main


class AdapterTests(unittest.TestCase):
    def test_python_words_match_native_golden(self):
        golden = (Path(__file__).resolve().parents[1] / 'validation/exchange/tensor_words.txt').read_text().splitlines()[1:]
        value = 1 | (1 << 63) | (1 << 71)
        known = (1 << 71) - 1
        wire = encode_tensor(value, known, [6, 12], ['row', 'col'])
        self.assertEqual([wire['value'], wire['known']], golden)
        self.assertEqual(decode_tensor(wire), (value, known, (6, 12), ('row', 'col')))

    def test_codec_rejects_padding_axis_and_packing(self):
        wire = encode_tensor(0x101, 1, (9,), ['lane'])
        for patch in ({'value': '0002000000000000'}, {'packing': 'big'}, {'bits': 10}, {'axes': ['a', 'a']}):
            with self.assertRaises(ValueError):
                decode_tensor({**wire, **patch})

    def test_signed_promotion_and_saturation(self):
        for width, next_width in ((16, 32), (32, 64)):
            half = 1 << (width - 1)
            self.assertEqual(StructuralCounter(half - 1, width).updated(1), StructuralCounter(half, next_width))
            self.assertEqual(StructuralCounter(-half, width).updated(-1), StructuralCounter(-half - 1, next_width))
        self.assertEqual(StructuralCounter().updated(1 << 65), StructuralCounter((1 << 63) - 1, 64))
        self.assertEqual(StructuralCounter().updated(-1 << 65), StructuralCounter(-(1 << 63), 64))
        self.assertEqual(StructuralCounter.from_dict({'value': -1, 'width': 32}).to_dict(), {'value': -1, 'width': 32})
        with self.assertRaises(ValueError):
            StructuralCounter(True, 16)


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.donor = VectorBooleanNet(seed=701, neurons=8)
        self.receiver = VectorBooleanNet(seed=977, neurons=8)
        self.meta = dict(source='explorer/701', license_name='user experiment', run='research-12', generation=1)

    def circuit(self):
        return circuit_candidate(self.donor, **self.meta, training_seed=271)

    def events(self):
        return experience_candidate(self.donor, **self.meta, scenario='moving',
                                    map_seed=271, episode_seed=507, size=8, steps=12)

    def test_recovers_circuit_behaviour_and_never_writes_receiver(self):
        original = self.receiver.export()
        imported = import_candidate(self.receiver, json.loads(json.dumps(self.circuit())))
        for i in range(20):
            obs = [(i * 113 >> j) & 1 for j in range(15)]
            self.assertEqual(imported.decide(obs), self.donor.decide(obs))
            self.assertEqual(imported.vectors, self.donor.vectors)
        self.assertEqual(original, self.receiver.export())

    def test_rejects_rehashed_incompatible_raw_contract(self):
        for field, invalid in [('seed', 42), ('hash', 'BLAKE3'), ('endian', 'big'), ('packing', 'msb'),
                               ('axes', ['angle', 'time']), ('sensor_encoding', 'broadcast'), ('limits', {})]:
            record = self.circuit()
            record['contract'][field] = invalid
            record['contract_sha256'] = digest(record['contract'])
            record['candidate_id'] = digest({k: v for k, v in record.items() if k != 'candidate_id'})
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                import_candidate(self.receiver, record)

    def test_named_training_events_survive_reencoding(self):
        record = self.events()
        self.receiver.reshape((64,))
        first = import_candidate(self.receiver, record)
        record['contract']['sensors'].reverse()
        record['contract']['seed'] = 150
        record['contract_sha256'] = digest(record['contract'])
        record['candidate_id'] = digest({k: v for k, v in record.items() if k != 'candidate_id'})
        self.assertEqual(first.export(), import_candidate(self.receiver, record).export())

    def test_rejects_tampering_and_leakage(self):
        record = self.events()
        for split in ('holdout', 'validation'):
            with self.assertRaisesRegex(ValueError, 'training'):
                import_candidate(self.receiver, {**record, 'split': split})
        with self.assertRaisesRegex(ValueError, 'integrity'):
            import_candidate(self.receiver, {**record, 'source': 'invented'})
        with self.assertRaisesRegex(ValueError, 'fresh seed'):
            evaluate_candidate(self.receiver, record, seed=271, episodes=1, steps=4)

    def test_null_candidate_fails_gate_without_mutating_baseline(self):
        original = self.donor.export()
        report, winner = evaluate_candidate(self.donor, self.circuit(), seed=991, episodes=1, steps=4)
        self.assertIsNone(winner)
        self.assertEqual(report['status'], 'rejected')
        self.assertEqual(report['baseline']['capture_rate'], report['candidate']['capture_rate'])
        self.assertEqual(report['sign_tail'], [1, 1])
        self.assertEqual(self.donor.export(), original)

    def test_cli_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            brain = Path(temp) / 'brain.json'
            before = json.dumps(self.donor.export())
            brain.write_text(before)
            with self.assertRaises(SystemExit):
                main(['export-circuit', str(brain), str(brain), '--source', 'EVA', '--run', 'run1',
                      '--generation', '1', '--license', 'user experiment', '--training-seed', '271'])
            self.assertEqual(brain.read_text(), before)
