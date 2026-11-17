"""Shared episode execution and scoring for V6 vector experiments."""
import hashlib
import json
import random
import statistics

from .progress_monitor import completion_metrics
from .world import World, DIRS
from .challenges import episode_specs


def genome_id(net):
    if hasattr(net, 'snapshot_id'):
        return net.snapshot_id()
    data = net.export()
    data.pop('state', None)
    data.pop('plastic_runtime', None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def rollout(net, scenario, seed, steps, record=False, observer=None):
    world = World(scenario['map'], seed=seed, visibility=scenario['visibility'],
                  prey_moves=scenario['prey_moves'])
    net.reset()
    frames = [{'t': 0, 'predator': list(world.predator), 'prey': list(world.prey)}] if record else None
    live = observer is not None and observer.begin_episode(net, scenario, seed)
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
        action = net.decide(observation)
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
            if record:
                frames.append(frame)
            if live:
                observer.add_frame(frame)
        if captured:
            break
        if len(trail) >= 20 and len(set(trail[-20:])) <= 2:
            stalled = True
            total -= 2
            break
    if live and hasattr(observer, 'end_episode'):
        observer.end_episode()
    return {'capture': int(world.captured), 'steps': world.steps, 'reward': total,
            'stall': int(stalled), 'blocked': blocked,
            'exploration': len(world.visited)/len(world._free_cells),
            'decoder_counts':dict(getattr(net,'decoder_counts',{}))}, frames


def evaluate_suite(net, suites, seed, split, episodes, steps, observer=None):
    rows = []
    task_metrics = {}
    for scenario in suites:
        scenario_rows = [rollout(net, task, sample_seed, steps, observer=observer)[0]
                         for task,sample_seed in episode_specs(scenario,seed,split,episodes)]
        task_metrics[scenario['name']] = {
            **completion_metrics(scenario_rows),
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
    metrics = {}
    all_rows = []
    for scenario in suites:
        task_rows = []
        for task,sample_seed in episode_specs(scenario,base_seed,split,episodes):
            world = World(task['map'], seed=sample_seed,
                          visibility=scenario['visibility'],
                          prey_moves=scenario['prey_moves'])
            rng = random.Random(sample_seed + 81282)
            reward_total = 0.
            blocked = 0
            stalled = False
            trail = []
            for _ in range(steps):
                if kind == 'random':
                    action = rng.randrange(4)
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
                if len(trail) >= 20 and len(set(trail[-20:])) <= 2:
                    stalled = True
                    reward_total -= 2
                    break
            task_rows.append({'capture': int(world.captured), 'steps': world.steps,
                              'reward': reward_total, 'stall': int(stalled),
                              'blocked': blocked,
                              'exploration': len(world.visited)/len(world._free_cells)})
        metrics[scenario['name']] = {
            **completion_metrics(task_rows),
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
        'note': ('Uses true prey coordinates; not restricted to EVA sensors' if kind == 'greedy'
                 else 'Uniform random cardinal actions'),
    }
