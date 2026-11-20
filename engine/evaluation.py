"""Shared episode execution and scoring for V6 vector experiments."""
import hashlib
import json
import random
import statistics
import math
import time

from .progress_monitor import completion_metrics
from .world import DIRS
from .challenge_worlds import make_world
from .challenges import episode_specs


def genome_id(net):
    if hasattr(net, 'snapshot_id'):
        return net.snapshot_id()
    data = net.export()
    data.pop('state', None)
    data.pop('plastic_runtime', None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def rollout(net, scenario, seed, steps, record=False, observer=None, measure=False):
    """One episode. V7: runs in the native kernel when available; the Python path
    (bit-identical) is used for recorded frames, sampled live frames and brains
    without a native twin."""
    from boolean_engine import native_backend as nb
    live_started = False
    if not record and nb.supports(net):
        if observer is not None:
            net.reset()
            live_started = observer.begin_episode(net, scenario, seed)
        if not live_started:
            return nb.rollout(net, scenario, seed, steps, measure), None
    world = make_world(scenario, seed)
    decision_ns = 0
    net.reset()
    frames = [{'t': 0, 'predator': list(world.predator), 'prey': list(world.prey)}] if record else None
    live = live_started or (observer is not None and observer.begin_episode(net, scenario, seed))
    if live:
        observer.add_frame({'t': 0, 'predator': list(world.predator), 'prey': list(world.prey),
                            'neural_state': list(net.state), 'initial': True})
    total = 0.
    blocked = 0
    stalled = False
    trail = []
    for _ in range(steps):
        observation = world.observe()
        before_state = list(net.state) if record or live else None
        if measure:
            begin = time.perf_counter_ns()
        action = net.decide(observation)
        if measure:
            decision_ns += time.perf_counter_ns() - begin
        previous = world.predator
        reward, captured = world.step(action)
        total += reward
        if world.predator == previous:
            blocked += 1
        trail.append(world.predator)
        if hasattr(net, 'transition'):
            net.transition(world.observe(), captured)
        if hasattr(net, 'feedback'):
            net.feedback(reward)
        if record or live:
            frame = {'t': world.steps, 'predator': list(world.predator),
                     'prey': list(world.prey), 'decision_position': list(previous),
                     'action': action, 'reward': reward, 'captured': captured,
                     'observation': list(observation), 'previous_state': before_state,
                     'neural_state': list(net.state)}
            frame.update(net.frame_data())
            if hasattr(world, 'world_events'):
                frame['world_events'] = world.world_events()
            if record:
                frames.append(frame)
            if live:
                observer.add_frame(frame)
        if captured:
            break
        if len(trail) >= 20 and len(set(trail[-20:])) <= 2 and getattr(world, 'due_tick', None) is None:
            stalled = True
            total -= 2
            break
    if live and hasattr(observer, 'end_episode'):
        observer.end_episode()
    return {'capture': int(world.captured), 'steps': world.steps, 'reward': total,
            'stall': int(stalled), 'blocked': blocked,
            'exploration': len(world.visited)/len(world._free_cells),
            'final_distance': sum(abs(a-b) for a,b in zip(world.predator, world.prey)),
            'obstacles': len(world.obstacles),
            **({'decision_ns': decision_ns} if measure else {}),
            'decoder_counts':dict(getattr(net,'decoder_counts',{}))}, frames


def capture_summary(rows):
    """Descriptive 95% Wilson interval conditional on this set of episodes."""
    n = len(rows)
    p = sum(row['capture'] for row in rows)/n
    z = 1.959963984540054
    scale = 1 + z*z/n
    center = (p + z*z/(2*n))/scale
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/scale
    times = [row['steps'] for row in rows if row['capture']]
    return {'capture_wilson95': [max(0.,center-half), min(1.,center+half)],
            'successful_median_steps': statistics.median(times) if times else None}


def sensor_greedy(observation, rng):
    """Only the 15 EVA inputs: prefer valid moves toward a visible target."""
    options = [i for i in range(4) if observation[5+i]]
    options.extend(4+i for i in range(4) if not observation[5+i] and observation[9+i])
    toward = [action for action in options if observation[1 + action % 4]]
    return rng.choice(toward or options or [0])


def evaluate_suite(net, suites, seed, split, episodes, steps, observer=None, record_at=None, recorded=None):
    """Episodes of every scenario on one adaptive brain, in order.

    record_at=(scenario index, episode index) records that episode's real frames into
    `recorded` (a list) as {'scenario', 'seed', 'frames'}; results are unchanged.
    """
    from boolean_engine import native_backend as nb
    rows = []
    task_metrics = {}
    plan = [(index, position, task, sample_seed) for index, scenario in enumerate(suites)
            for position, (task, sample_seed) in enumerate(episode_specs(scenario, seed, split, episodes))]

    def recorded_episode(task, sample_seed):
        row, frames = rollout(net, task, sample_seed, steps, record=True, observer=observer)
        if recorded is not None:
            recorded.append({'scenario': task, 'seed': sample_seed, 'frames': frames})
        return row

    batched = {}
    if observer is None and nb.supports(net):
        # V7: the suite runs in native batches; a recorded episode uses the identical Python path.
        cut = next((i for i, item in enumerate(plan) if (item[0], item[1]) == record_at), None)
        pieces = [plan] if cut is None else [plan[:cut], plan[cut:cut+1], plan[cut+1:]]
        results = []
        for part, piece in enumerate(pieces):
            if not piece:
                continue
            if cut is not None and part == 1:
                results.append(recorded_episode(piece[0][2], piece[0][3]))
            else:
                results.extend(nb.run_episodes(net, [(task, s) for _, _, task, s in piece], steps))
        for item, row in zip(plan, results):
            batched.setdefault(item[0], []).append(row)
    else:
        for index, position, task, sample_seed in plan:
            row = (recorded_episode(task, sample_seed) if (index, position) == record_at else
                   rollout(net, task, sample_seed, steps, observer=observer)[0])
            batched.setdefault(index, []).append(row)
    for index, scenario in enumerate(suites):
        scenario_rows = batched.get(index, [])
        task_metrics[scenario['name']] = {
            **completion_metrics(scenario_rows),
            **capture_summary(scenario_rows),
            'mean_final_distance': statistics.mean(r['final_distance'] for r in scenario_rows),
            'mean_obstacles': statistics.mean(r['obstacles'] for r in scenario_rows),
            'decoder_counts': {key: sum(r['decoder_counts'].get(key,0) for r in scenario_rows) for key in net.decoder_counts},
            'capture_rate': statistics.mean(row['capture'] for row in scenario_rows),
            'mean_steps': statistics.mean(row['steps'] for row in scenario_rows),
            'stall_rate': statistics.mean(row['stall'] for row in scenario_rows),
            'stall_count': sum(row['stall'] for row in scenario_rows),
            'blocked_actions': statistics.mean(row['blocked'] for row in scenario_rows),
            'exploration': statistics.mean(row['exploration'] for row in scenario_rows),
            'mean_reward': statistics.mean(row['reward'] for row in scenario_rows),
            'paired_outcomes': {
                'episodes': len(scenario_rows),
                'captures': sum(row['capture'] << i for i,row in enumerate(scenario_rows)),
                'stalls': sum(row['stall'] << i for i,row in enumerate(scenario_rows)),
            },
        }
        rows.extend(scenario_rows)
    metrics = {
        **completion_metrics(rows),
        'capture_rate': statistics.mean(row['capture'] for row in rows),
        'mean_steps': statistics.mean(row['steps'] for row in rows),
        'mean_reward': statistics.mean(row['reward'] for row in rows),
        'stall_rate': statistics.mean(row['stall'] for row in rows),
        'stall_count': sum(row['stall'] for row in rows),
        'blocked_actions': statistics.mean(row['blocked'] for row in rows),
        'exploration': statistics.mean(row['exploration'] for row in rows),
        'active_neurons': sum(node.active for node in net.nodes),
        'total_neurons': len(net.nodes),
        'executed_neurons': len(net.computed_nodes),
        'vector_width':net.width,
        'decoder_counts':{k:sum(r['decoder_counts'].get(k,0) for r in rows) for k in net.decoder_counts},
        'min_task_capture': min(metrics['capture_rate'] for metrics in task_metrics.values()),
        'by_scenario': task_metrics,
    }
    return metrics


def rank(metrics):
    return (metrics['capture_rate'], metrics['min_task_capture'],
            metrics['exploration'], -metrics['stall_rate'],
            -metrics['blocked_actions'], metrics['mean_reward'],
            -metrics['active_neurons'])


def baseline_suite(kind, suites, base_seed, split, episodes, steps):
    """Diagnostic random and full-information Greedy baselines, not EVA training."""
    if kind not in ('random','sensor_greedy','greedy'):
        raise ValueError(f'Unknown baseline: {kind}')
    metrics = {}
    all_rows = []
    for scenario in suites:
        task_rows = []
        for task,sample_seed in episode_specs(scenario,base_seed,split,episodes):
            world = make_world(task, sample_seed)
            rng = random.Random(sample_seed + 81282)
            reward_total = 0.
            blocked = 0
            stalled = False
            trail = []
            for _ in range(steps):
                if kind == 'random':
                    action = rng.randrange(4)
                elif kind == 'sensor_greedy':
                    action = sensor_greedy(world.observe(), rng)
                else:
                    options = []
                    for index, (dx, dy) in enumerate(DIRS):
                        position = (world.predator[0] + dx, world.predator[1] + dy)
                        if world.free(position):
                            distance = abs(position[0]-world.prey[0]) + abs(position[1]-world.prey[1])
                            options.append((distance, index))
                    action = min(options)[1] if options else 0
                previous = world.predator
                reward, captured = world.step(action)
                reward_total += reward
                if world.predator == previous:
                    blocked += 1
                trail.append(world.predator)
                if captured:
                    break
                if len(trail) >= 20 and len(set(trail[-20:])) <= 2 and getattr(world, 'due_tick', None) is None:
                    stalled = True
                    reward_total -= 2
                    break
            task_rows.append({'capture': int(world.captured), 'steps': world.steps,
                              'reward': reward_total, 'stall': int(stalled),
                              'blocked': blocked,
                              'exploration': len(world.visited)/len(world._free_cells)})
        metrics[scenario['name']] = {
            **completion_metrics(task_rows),
            **capture_summary(task_rows),
            'capture_rate': statistics.mean(row['capture'] for row in task_rows),
            'mean_steps': statistics.mean(row['steps'] for row in task_rows),
            'stall_rate': statistics.mean(row['stall'] for row in task_rows),
            'blocked_actions': statistics.mean(row['blocked'] for row in task_rows),
        }
        all_rows.extend(task_rows)
    return {
        **completion_metrics(all_rows),
        'capture_rate': statistics.mean(row['capture'] for row in all_rows),
        'mean_steps': statistics.mean(row['steps'] for row in all_rows),
        'stall_rate': statistics.mean(row['stall'] for row in all_rows),
        'blocked_actions': statistics.mean(row['blocked'] for row in all_rows),
        'exploration': statistics.mean(row['exploration'] for row in all_rows),
        'mean_reward': statistics.mean(row['reward'] for row in all_rows),
        'by_scenario': metrics,
        'note': {'greedy': 'Privileged reference using true prey coordinates',
                 'sensor_greedy':'Greedy with the same 15 local sensors as EVA',
                 'random':'Uniform random cardinal actions'}[kind],
    }
