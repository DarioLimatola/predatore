"""Verifica la LUT Rust su episodi V6.05 e misura anche il costo del rollout completo."""
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boolean_engine import BinaryVectorNetwork, pack
from engine.world import World

GOLDEN = ROOT / 'validation' / 'p2' / 'golden.json'
BINARY = ROOT / 'native' / 'target' / 'release' / ('predatore-lut-probe.exe' if os.name == 'nt' else 'predatore-lut-probe')
END = 2**32 - 1


def controller(case):
    net = BinaryVectorNetwork(case['network_seed'], neurons=case['neurons'], internal_ticks=case['ticks'])
    net.associative_enabled = net.plasticity_enabled = False
    net.memory_enabled = case['memory']
    net.sensor_encoding = case['encoding']
    if case.get('mutated_masks'):
        rng = random.Random(1281)
        for rule in net.rules:
            rule['write'] = rng.getrandbits(64)
            rule['lut'] = [rng.getrandbits(64) for _ in range(4)]
    return net


def header(net, *, rows, horizon, start=0, table=False, trace=False):
    """Network is reset at episode start. Reject features outside native semantics."""
    if (net.width != 64 or net.sensors > 32 or net.associative_enabled or
            net.plasticity_enabled or net.collect_experience or getattr(net, 'relation_credit', None) is not None):
        raise ValueError('P2 supports only a frozen circuit decoder with 64 lanes and <=32 sensors')
    indices = net._output_ancestors() if net.sparse_enabled else [i for i, n in enumerate(net.nodes) if n.active]
    flags = int(trace) | (int(net.sensor_encoding == 'broadcast') << 1) | (int(net.memory_enabled) << 2)
    data = bytearray(struct.pack('<8s11I', b'PDLUT1', 64, net.sensors, len(net.nodes), len(indices),
                                 net.internal_ticks, net.action_bits, flags, table, rows, horizon, start))
    for idx in indices:
        node, rule = net.nodes[idx], net.rules[idx]
        for side in ('a', 'b'):
            if ((net.memory_enabled and rule.get('delay_' + side, 0)) or
                    (net.tensor_enabled and rule.get('operator_' + side, 'rotate') != 'rotate')):
                raise ValueError('unsupported operand in P2')
        data.extend(struct.pack('<6I5Q', idx, node.a, node.b, rule['hold'] + 1, rule['shift_a'] % 64,
                                rule['shift_b'] % 64, rule['write'], *rule['lut']))
    for node, lane in zip(net.outputs, net.output_lanes):
        data.extend(struct.pack('<II', node, lane))
    return data


def static_table(case):
    """Compile the finite observation/transition table by asking the real Python world."""
    if case['moving']:
        raise ValueError('P2 table requires a stationary prey')
    world = World(case['map'], seed=case['seed'], visibility=case['visibility'], prey_moves=False)
    cells = {p: 2*i for i, p in enumerate(world._free_cells)}
    start = cells[world.predator]
    rows = bytearray()
    for pos in cells:
        for collided in range(2):
            world.predator, world.collisions = pos, collided
            obs = pack(world.observe())
            edges = []
            for action in range(8):
                world.predator, world.collisions = pos, collided
                _, captured = world.step(action)
                edges.append(END if captured else cells[world.predator] + int(world.collisions > 0))
            rows.extend(struct.pack('<Q8I', obs, *edges))
    return start, len(cells)*2, rows


def packet(net, case, observations=None, trace=False):
    if observations is not None:
        n = len(observations)
        return header(net, rows=n, horizon=n, trace=trace) + struct.pack(f'<{n}Q', *observations)
    start, count, rows = static_table(case)
    return header(net, rows=count, horizon=case['steps'], start=start, table=True, trace=trace) + rows


def live_episode(case, net=None, actions=None, trace=False):
    """Actual world and stop conditions. Used identically for scoring either controller."""
    world = World(case['map'], seed=case['seed'], visibility=case['visibility'], prey_moves=case['moving'])
    if net is not None:
        net.reset()
    observations, payload, trajectory = [], bytearray(), []
    score, trail = 0.0, []
    for t in range(case['steps']):
        bits = world.observe()
        action = net.decide(bits) if actions is None else actions[t]
        reward, captured = world.step(action)
        if net is not None:
            net.transition(world.observe(), captured)
            net.feedback(reward)
        if trace:
            observations.append(pack(bits))
            payload.append(action)
            if net is not None:
                payload.extend(struct.pack(f'<{len(net.nodes)}Q', *net.vectors))
            trajectory.append([*world.predator, *world.prey, reward, world.collisions, captured])
        score += reward
        trail.append(world.predator)
        if captured:
            break
        if len(trail) >= 20 and len(set(trail[-20:])) <= 2:
            score -= 2
            break
    return observations, payload, trajectory, (score, world.steps, world.collisions, world.captured)


def digest_trajectory(record):
    return hashlib.sha256(json.dumps(record, separators=(',', ':')).encode()).hexdigest()


def verify(binary, cases):
    decisions = 0
    for case in cases:
        net = controller(case)
        obs, states, trajectory, metrics = live_episode(case, net, trace=True)
        if (hashlib.sha256(states).hexdigest() != case['states_sha256'] or
                digest_trajectory(trajectory) != case['trajectory_sha256'] or
                list(metrics) != case['metrics'] or obs != case['observations']):
            raise AssertionError(f"{case['name']} drift from V6.05")
        tape = packet(net, case, observations=obs, trace=True)
        native = subprocess.run([binary], input=tape, capture_output=True, check=True).stdout
        if native != states:
            raise AssertionError(f"{case['name']} action/vector parity failed")
        if not case['moving']:
            full = subprocess.run([binary], input=packet(net, case), capture_output=True, check=True).stdout
            _, _, replay, result = live_episode(case, actions=full, trace=True)
            if replay != trajectory or result != metrics:
                raise AssertionError(f"{case['name']} closed-loop divergence")
        decisions += len(obs)
    return decisions


def timed(binary, cases, reps):
    rows = []
    for rep in range(reps):
        for case in cases:
            if case['moving']:
                continue
            net = controller(case)
            times = {}
            for backend in (('python', 'rust') if rep % 2 == 0 else ('rust', 'python')):
                t0 = time.perf_counter()
                if backend == 'python':
                    _, _, _, ref = live_episode(case, net)
                    times['python_seconds'] = time.perf_counter() - t0
                else:
                    data = packet(net, case)
                    t1 = time.perf_counter()
                    actions = subprocess.run([binary], input=data, capture_output=True, check=True).stdout
                    t2 = time.perf_counter()
                    _, _, _, got = live_episode(case, actions=actions)
                    t3 = time.perf_counter()
                    times.update(rust_total_seconds=t3-t0, prepare_seconds=t1-t0, subprocess_seconds=t2-t1,
                                 score_seconds=t3-t2, request_bytes=len(data))
            if ref != got:
                raise AssertionError('scoring mismatch')
            rows.append(dict(repetition=rep+1, case=case['name'], steps=ref[1], **times))
    return rows


def resource_sample(backend, binary, cases, repetitions):
    """Measure driver and maximum single child RSS separately in a fresh process."""
    try:
        import resource
    except ImportError:
        resource = None
    cpu0 = time.process_time()
    for _ in range(repetitions):
        for case in cases:
            if case['moving']:
                continue
            net = controller(case)
            if backend == 'python':
                result = live_episode(case, net)[-1]
            else:
                data = packet(net, case)
                actions = subprocess.run([binary], input=data, capture_output=True, check=True).stdout
                result = live_episode(case, actions=actions)[-1]
            if list(result) != case['metrics']:
                raise AssertionError('resource pass failed parity')
    measured = {'driver_cpu_seconds': time.process_time()-cpu0,
                'child_cpu_seconds': None, 'driver_peak_mib': None, 'max_child_peak_mib': None}
    if resource:
        parent = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        scale = 1024**2 if sys.platform == 'darwin' else 1024
        measured.update(driver_peak_mib=parent.ru_maxrss/scale,
                        max_child_peak_mib=child.ru_maxrss/scale, child_cpu_seconds=child.ru_utime+child.ru_stime)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=BINARY)
    parser.add_argument('--repetitions', type=int, default=9)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resource-worker', choices=('python', 'rust'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error('compile first with cargo build --release --locked --manifest-path native/Cargo.toml')
    if args.repetitions < 1:
        parser.error('positive repetition count required')
    cases = json.loads(GOLDEN.read_text())['cases']
    if args.resource_worker:
        print(json.dumps(resource_sample(args.resource_worker, args.binary, cases, args.repetitions)))
        return
    decisions = verify(args.binary, cases)
    rows = timed(args.binary, cases, args.repetitions)
    batch = {key: statistics.median(sum(row[key] for row in rows if row['repetition'] == rep)
                                    for rep in range(1, args.repetitions+1))
             for key in ('python_seconds', 'rust_total_seconds', 'prepare_seconds', 'subprocess_seconds', 'score_seconds')}
    info = {'python': sys.version, 'system': platform.platform(), 'cases': len(cases), 'decisions': decisions,
            'closed_loop_cases': sum(not c['moving'] for c in cases), 'repetitions': args.repetitions,
            'median_batch': batch, 'whole_batch_speedup': batch['python_seconds']/batch['rust_total_seconds'],
            'rows': rows}
    info['resources'] = {
        name: json.loads(subprocess.check_output([sys.executable, __file__, '--binary', str(args.binary),
                                                 '--repetitions', str(args.repetitions), '--resource-worker', name]))
        for name in ('python', 'rust')}
    if args.output:
        args.output.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
