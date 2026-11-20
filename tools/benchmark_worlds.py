"""Co-adapt independent predator/prey genomes and test transfer on unseen maps.

Run from the project root: python -m tools.benchmark_worlds --output result.json
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from boolean_engine import VectorBooleanNet
from engine.challenges import build_scenario
from engine.challenge_worlds import make_world
from engine.evaluation import capture_summary, rollout, sensor_greedy
from engine.progress_monitor import completion_metrics

TASKS = ('maze_novel', 'hidden', 'delayed_reward', 'intelligent_prey')


def offspring(parent, seed):
    child = VectorBooleanNet.restore(parent)
    child.recovery_mutate(seed, max_neurons=40, allow_growth=False, allow_reactivate=False)
    return child.export()


def duel(predator, prey, schedule, steps):
    agent = VectorBooleanNet.restore(predator)
    rows = []
    for scenario, seed in schedule:
        row, _ = rollout(agent, {**scenario, 'prey_genome': prey}, seed, steps)
        rows.append(row)
    return (sum(r['capture'] for r in rows), -sum(r['steps'] for r in rows if r['capture'])), agent.export()


def coadapt(seed, generations, episodes, steps):
    rng = random.Random(seed)
    predator = VectorBooleanNet(seed, neurons=32, shape=(8,)).export()
    prey = VectorBooleanNet(seed+1, neurons=32, shape=(8,)).export()
    initial_predator = predator
    initial_prey = prey
    history = []
    for gen in range(generations):
        schedule = [(build_scenario('intelligent_prey', 8, rng.getrandbits(63)),
                     rng.getrandbits(63)) for _ in range(episodes)]
        pred_candidate = offspring(predator, rng.getrandbits(63))
        prey_candidate = offspring(prey, rng.getrandbits(63))
        incumbent, learned = duel(predator, prey, schedule, steps)
        challenger, learned_candidate = duel(pred_candidate, prey, schedule, steps)
        against_evader, _ = duel(predator, prey_candidate, schedule, steps)
        # Each candidate faces the previous generation's opponent on identical
        # starts. Equality keeps the incumbent, preserving its lineage.
        predator = learned_candidate if challenger > incumbent else learned
        if against_evader < incumbent:
            prey = prey_candidate
        history.append({'generation': gen+1, 'train_map_episode_seeds':
                        [[scenario['map']['seed'], s] for scenario,s in schedule],
                        'incumbent_score': incumbent, 'hunter_candidate_score': challenger,
                        'hunter_candidate_adopted': challenger > incumbent,
                        'evader_candidate_score_against_hunter': against_evader,
                        'evader_candidate_adopted': against_evader < incumbent})
    return initial_predator, predator, initial_prey, prey, history


def run(seed=83, generations=3, train_episodes=5, maps=3, episodes=4, steps=36):
    initial, trained, native_prey, trained_prey, history = coadapt(seed, generations, train_episodes, steps)
    rows = []
    report = {'protocol': {'seed': seed, 'training_size': 8, 'generations': generations,
              'training_episodes_per_candidate':train_episodes,
              'training_scenario':'intelligent_prey', 'selection_split':'train',
              'evaluation_split':'holdout', 'holdout_maps_per_size':maps,
              'episodes_per_map':episodes, 'steps':steps, 'sizes':[8,12],
              'agent_policy':'frozen at start of every test episode',
              'evader_policy':'learns online within its own isolated episode',
              'capture_interval':'descriptive Wilson 95% pooled across starts; starts share maps'},
              'training':history, 'results':rows}
    for size in (8, 12):
        for task_index, task in enumerate(TASKS):
            raw = {}
            for map_index in range(maps):
                # Explicit non-overlapping namespace with the training seeds
                # (non-negative getrandbits); same held-out maps for all policies.
                map_seed = -(1 + seed*100000 + size*1000 + task_index*100 + map_index)
                scenario = build_scenario(task, size, map_seed)
                if task == 'intelligent_prey':
                    scenario['prey_genome'] = trained_prey
                for episode in range(episodes):
                    episode_seed = map_seed*1024-episode
                    policies = [('initial', initial, scenario), ('trained8', trained, scenario)]
                    if task == 'intelligent_prey':
                        policies.append(('trained8_initial_evader', trained,
                                         {**scenario,'prey_genome':native_prey}))
                    for actor, brain, arena in policies:
                        agent = VectorBooleanNet.restore(brain)
                        agent.plasticity_enabled = False
                        row,_ = rollout(agent, arena, episode_seed, steps, measure=True)
                        raw.setdefault(actor,[]).append({'map_seed':map_seed,'seed':episode_seed,**row})
                    # Sharing explicit episode specs makes the random and greedy
                    # starts identical to the neural agents' starts.
                    for actor in ('random8','sensor_greedy'):
                        rng = random.Random(episode_seed+81282)
                        world = make_world(scenario, episode_seed)
                        trail = []
                        for _ in range(steps):
                            action = rng.randrange(8) if actor=='random8' else sensor_greedy(world.observe(), rng)
                            _, captured = world.step(action)
                            trail.append(world.predator)
                            if captured or (len(trail)>=20 and len(set(trail[-20:]))<=2
                                            and getattr(world,'due_tick',None) is None):
                                break
                        raw.setdefault(actor,[]).append({'map_seed':map_seed,'seed':episode_seed,
                            'capture':int(world.captured),'steps':world.steps,'blocked':world.collisions})
            for actor, episodes_data in raw.items():
                stats = {**completion_metrics(episodes_data), **capture_summary(episodes_data)}
                stats['capture_rate'] = stats['capture_count']/len(episodes_data)
                stats['mean_collisions'] = statistics.mean(item['blocked'] for item in episodes_data)
                if 'decision_ns' in episodes_data[0]:
                    stats['mean_decision_microseconds'] = (sum(item['decision_ns'] for item in episodes_data)
                                                          /sum(item['steps'] for item in episodes_data)/1000)
                rows.append({'size':size,'scenario':task,'actor':actor,**stats,'episodes':episodes_data})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed',type=int,default=83)
    p.add_argument('--generations',type=int,default=3)
    p.add_argument('--train-episodes',type=int,default=5)
    p.add_argument('--maps',type=int,default=3)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--steps',type=int,default=36)
    args = p.parse_args()
    if args.seed < 0:
        p.error('seed must be nonnegative')
    if min(args.generations,args.train_episodes,args.maps,args.episodes,args.steps) < 1:
        p.error('budgets must be positive')
    result=run(args.seed,args.generations,args.train_episodes,args.maps,args.episodes,args.steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
