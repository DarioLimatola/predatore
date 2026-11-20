"""The Python oracle always runs. Installed Rust binaries must match its immutable corpus."""
import hashlib
import json
import subprocess
import unittest

from tools.benchmark_native import BINARY, GOLDEN, controller, digest_trajectory, header, live_episode, verify


class NativeCircuitProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(GOLDEN.read_text())['cases']

    def test_python_matches_v605_golden(self):
        for case in self.cases:
            with self.subTest(case=case['name']):
                obs, trace, trail, metrics = live_episode(case, controller(case), trace=True)
                self.assertEqual(obs, case['observations'])
                self.assertEqual(hashlib.sha256(trace).hexdigest(), case['states_sha256'])
                self.assertEqual(digest_trajectory(trail), case['trajectory_sha256'])
                self.assertEqual(list(metrics), case['metrics'])

    def test_refuses_unsupported_decoder_and_delay(self):
        net = controller(self.cases[0])
        net.associative_enabled = True
        with self.assertRaises(ValueError):
            header(net, rows=1, horizon=1)
        net.associative_enabled = False
        net.rules[net.outputs[0]]['delay_a'] = 1
        with self.assertRaises(ValueError):
            header(net, rows=1, horizon=1)

    @unittest.skipUnless(BINARY.is_file(), 'compile the optional Rust P2 probe to compare the native executable')
    def test_real_binary_golden_and_live(self):
        self.assertEqual(verify(BINARY, self.cases), 289)
        result = subprocess.run([BINARY], input=b'PDLUT1', capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')
