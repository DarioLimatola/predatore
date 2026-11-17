"""Evolving explorers, one learning EVA, and persistent SMITH supervision.

Training measures real adaptive episodes. Independent evaluation copies learn
locally, then are discarded: validation/test experiences never enter SMITH.
Each generation is a recoverable local-then-global transaction.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import copy
import json
import math
import os
import random
import re
import statistics
import time

from .vector_boolean import VectorBooleanNet
from .contextual import distill_into
from .settings import Limits
from .paired_evidence import paired_capture_evidence
from . import smith
from .boolean_net import BooleanNet
from .challenges import prepare_suites, seeds_for, training_generation
from .evaluation import rollout, evaluate_suite, baseline_suite, rank
from .progress_monitor import population_completion
from .vault import (atomic_json, append_jsonl, read_json, save_checkpoint,
                    latest_checkpoint_path, reconcile_evolution_logs, seal, SCHEMA)


@contextmanager
def profile_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory/'writer.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('SMITH occupato: un altro training usa questo profilo') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('SMITH occupato: un altro training usa questo profilo') from exc
        yield
    finally:
        handle.close()


def profile_path(root, name=Limits.DEFAULT_SMITH_PROFILE):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('Profilo SMITH: lettere, numeri, _ o - (max 64)')
    return Path(root)/'EvolutionVault'/'SMITH'/name


def load_profile(root, name=Limits.DEFAULT_SMITH_PROFILE):
    path = profile_path(root, name)
    latest = latest_checkpoint_path(path)
    if latest:
        state = read_json(latest)
        if state.get('schema') != Limits.PROFILE_SCHEMA:
            raise ValueError('Formato non supportato: usare una cartella V6.05 pulita.')
        return state
    if (path/'latest.json').exists():
        raise ValueError('Memoria SMITH non leggibile: recuperare un checkpoint valido')
    return None


def evaluation_copy(net, learning=True):
    """Disposable evaluation object, never a second persistent agent."""
    result = BooleanNet.restore(net.export())
    result.reset()
    result.plasticity_enabled = learning
    result.collect_experience = False
    result.circuit_plasticity_enabled = False
    return result


def assessment(net, suites, seed, count, steps, split='validation'):
    return evaluate_suite(evaluation_copy(net), suites, seed, split, count, steps)


class ProbeCache:
    """Deterministic isolated probes, keyed by complete initial learned state."""
    def __init__(self):
        self.rows = OrderedDict()
        self.hits = 0

    def assess(self, net, task):
        key = (net.snapshot_id(), json.dumps(task, sort_keys=True))
        if key in self.rows:
            self.hits += 1
            self.rows.move_to_end(key)
            return self.rows[key]
        value = assessment(net, task['suites'], task['seed'], task['episodes'], task['steps'])
        self.rows[key] = value
        if len(self.rows) > Limits.PROBE_CACHE_ENTRIES:
            self.rows.popitem(last=False)
        return value


def quality(rows):
    return (sum(r['capture_rate'] for r in rows), -sum(r['stall_rate'] for r in rows),
            sum(r['mean_reward'] for r in rows))


def probe_scores(net, tasks, cache=None):
    # One adaptive copy per task, preserving learning across that task's episodes.
    return [(cache.assess(net, task) if cache else
             assessment(net, task['suites'], task['seed'], task['episodes'], task['steps']))
            for task in tasks]


def selection_key(rows, neurons):
    """Current run gets half the capture weight; past tasks share the other half.

    Exact rational capture/stall counts avoid floating-point tolerance decisions.
    This is an empirical ranking, not a statistical guarantee of non-regression.
    A scenario loss can be offset by a larger gain; all losses remain visible.
    """
    def rate(row, field):
        return Fraction(row[field], row['episode_count'])
    current = rows[-1]
    def balanced(field):
        now = rate(current, field)
        past = sum((rate(row, field) for row in rows[:-1]), Fraction()) / max(1, len(rows)-1)
        return (now+past)/2 if len(rows)>1 else now
    reward = (current['mean_reward'] + statistics.mean(r['mean_reward'] for r in rows[:-1]))/2 if len(rows)>1 else current['mean_reward']
    return (balanced('capture_count'), rate(current, 'capture_count'),
            -balanced('stall_count'), reward, -neurons)


def select_transfer(eva, proposals, tasks, cache=None):
    before = probe_scores(eva, tasks, cache)
    baseline_key = selection_key(before, len(eva.nodes))
    best_key, selected, selected_rows = baseline_key, None, before
    decisions = []
    for label, candidate, taught in proposals:
        identity = candidate.snapshot_id()
        changed = identity != eva.snapshot_id()
        rows = probe_scores(candidate, tasks, cache) if changed else before
        key = selection_key(rows, len(candidate.nodes))
        eligible = changed and key >= baseline_key
        # Prefer the first equal candidate (contextual student), but retain a better one.
        if eligible and (selected is None or key > best_key):
            best_key, selected, selected_rows = key, (label, candidate, taught), rows
        scenarios = []
        for task, old, new in zip(tasks, before, rows):
            for name in sorted(old['by_scenario']):
                a, b = old['by_scenario'][name], new['by_scenario'][name]
                scenarios.append({'task_run': task['run'], 'scenario': name,
                    'capture_delta': b['capture_count']-a['capture_count'],
                    'stall_delta': b['stall_count']-a['stall_count'],
                    'reward_delta': b['mean_reward']-a['mean_reward'],
                    'paired_capture_evidence': paired_capture_evidence(a, b)})
        decisions.append({'proposal': label, 'eligible': eligible, 'selected': False,
                          'candidate_id': identity, 'neurons': len(candidate.nodes),
                          'score_before': [str(v) for v in baseline_key],
                          'score_after': [str(v) for v in key],
                          'reason': 'unchanged' if not changed else 'competitive' if eligible else 'lower_balanced_score',
                          'diagnostics': {'complete': True, 'scenarios': scenarios}})
    if selected:
        next(d for d in decisions if d['proposal']==selected[0])['selected'] = True
    return selected, best_key>baseline_key, decisions


def fit_topology(net, minimum, maximum, seed):
    """Apply explicit user bounds; learned action memories survive resizing."""
    while len(net.nodes)>maximum:
        net.remove_neuron(len(net.nodes)-1)
    while len(net.nodes)<minimum:
        if not net.grow(seed+len(net.nodes)*7919, maximum):
            raise ValueError('Minimo moduli oltre il budget di stato')
    # Training evolves physical graph size; inactive modules are diagnostic interventions.
    if any(not node.active for node in net.nodes):
        raise ValueError('Il circuito contiene moduli inattivi non supportati')
    net.reset()


def choose_teachers(scored, limit=Limits.TEACHER_CANDIDATES):
    chosen = [scored[0]]
    for name in sorted(scored[0][0]['by_scenario']):
        best = max(scored, key=lambda pair: quality([pair[0]['by_scenario'][name]]))
        if all(best[1] is not pair[1] for pair in chosen):
            chosen.append(best)
    for pair in scored:
        if len(chosen)>=limit:
            break
        if all(pair[1] is not old[1] for old in chosen):
            chosen.append(pair)
    return chosen[:limit]


def build_transfer_proposals(eva, teachers, events, *, contextual=True,
                             focus_terminal=False, distillation_report=None):
    """Distill selected teachers; offer intact circuits when compression loses skill."""
    proposals = []
    if contextual:
        student = evaluation_copy(eva)
        report = distill_into(student, events, focus_terminal=focus_terminal)
        if distillation_report is not None:
            distillation_report.update(report)
        proposals.append(('contextual_distillation', student, report['contexts_taught']))
        merged = evaluation_copy(teachers[0][1])
        merge_report = distill_into(merged, events, focus_terminal=focus_terminal)
        if merge_report['contexts_taught']:
            proposals.append(('teacher_distilled', merged, merge_report['contexts_taught']))
    for index, (_, teacher) in enumerate(teachers):
        proposals.append((f'teacher_{index+1}', evaluation_copy(teacher), 0))
    for index, candidate in enumerate(teachers[0][1].local_circuit_candidates(Limits.LOCAL_CANDIDATES)):
        net = evaluation_copy(teachers[0][1])
        net.apply_local_circuit_candidate(candidate)
        net.reset()
        proposals.append((f'local_circuit_{index+1}', net, 0))
    unique, seen = [], set()
    for label, net, count in proposals:
        identity = net.snapshot_id()
        if identity not in seen:
            seen.add(identity)
            unique.append((label, net, count))
    return unique


def select_events(old, new, capacity, seed):
    """Bounded deterministic reservoir, stratified by contributing agent."""
    groups = {}
    for event in {e['event_id']: e for e in old+new}.values():
        groups.setdefault(event['source'], []).append(event)
    rng = random.Random(seed)
    for events in groups.values():
        rng.shuffle(events)
    sources = sorted(groups)
    rng.shuffle(sources)
    selected = []
    while sources and len(selected)<capacity:
        remaining = []
        for source in sources:
            if len(selected)>=capacity:
                break
            selected.append(groups[source].pop())
            if groups[source]:
                remaining.append(source)
        sources = remaining
    return selected


def retain_explorers(scored):
    return [{'id': m['source'], 'brain': net.export(), 'training_capture': m['capture_rate']}
            for m, net in choose_teachers(scored, Limits.EXPLORER_RETENTION)]


def behavior(m):
    return [v for name in sorted(m['by_scenario']) for v in
            (m['by_scenario'][name]['capture_rate'], m['by_scenario'][name]['exploration'],
             m['by_scenario'][name]['stall_rate'])]


def novelty(desc, archive):
    if not archive:
        return 1.
    distances = sorted(sum(abs(a-b) for a,b in zip(desc,d))/len(desc) for d in archive)
    return statistics.mean(distances[:min(Limits.NOVELTY_NEIGHBORS,len(distances))])


def train_actor(net, source, suites, c, count, sample_seed, observer=None):
    sample, seen = [], 0
    rng = random.Random(sample_seed)
    quota = max(Limits.MIN_AGENT_SAMPLE, c.experience_capacity//(c.population+1))
    def collect(event):
        nonlocal seen
        seen += 1
        event = {**event, 'source': source, 'event_id': f'{source}/t{seen}'}
        if len(sample)<quota:
            sample.append(event)
        else:
            index = rng.randrange(seen)
            if index<quota:
                sample[index] = event
            elif event['terminal_success']:
                sample[-1] = event
    net.plasticity_enabled = True
    net.collect_experience = True
    net.experience_sink = collect
    net.circuit_plasticity_enabled = c.circuit_plasticity_enabled
    net.clear_circuit_evidence()
    try:
        metrics = evaluate_suite(net, suites, c.seed, 'train', count, c.steps, observer=observer)
    finally:
        net.collect_experience = False
        net.experience_sink = None
    return {**metrics, 'source': source}, sample, seen


def run_vector(root, config, resume=False, progress=None, should_stop=None, observer=None, monitor=None):
    directory = profile_path(root, config.smith_profile)
    with profile_lock(directory):
        return _run(root, config, resume, progress, should_stop, observer, monitor, directory)


def _run(root, c, resume, progress, should_stop, observer, monitor, directory):
    if not (Limits.MIN_POPULATION<=c.population<=Limits.MAX_POPULATION and
            1<=c.elite<c.population and 1<=c.generations<=Limits.MAX_GENERATIONS and
            1<=c.episodes<=Limits.MAX_EPISODES and 1<=c.steps<=Limits.MAX_STEPS and
            Limits.MIN_WORLD_SIZE<=c.size<=Limits.MAX_WORLD_SIZE and
            Limits.MIN_NEURONS<=c.minimum_active_neurons<=c.initial_neurons<=c.max_neurons<=Limits.MAX_NEURONS and
            c.validation_episodes>=1 and c.holdout_episodes>=1 and
            Limits.MIN_WIDTH<=c.vector_width<=c.max_vector_width<=Limits.MAX_WIDTH and c.memory_capacity>=Limits.MIN_MEMORY and
            c.experience_capacity>=c.population+1 and c.retention_tasks>=1):
        raise ValueError('Configurazione non valida: 32 <= min <= base <= max <= 4096 e budget positivi')
    shape = tuple(int(x) for x in c.vector_shape.split(',')) if c.vector_shape else (Limits.DEFAULT_SHAPE if c.vector_width==Limits.DEFAULT_WIDTH else (c.vector_width,))
    if not shape or len(shape)>Limits.MAX_AXES or any(x<=0 for x in shape) or math.prod(shape)!=c.vector_width:
        raise ValueError('Il prodotto della forma dei moduli deve essere uguale a vector_width')
    folder = Path(root)/'EvolutionVault'/'experiments'/f'EXP_seed{c.seed}_size{c.size}'
    existing = load_profile(root, c.smith_profile)
    cp = latest_checkpoint_path(folder/'checkpoints')
    scored = []
    if resume:
        if cp is None:
            raise FileNotFoundError('Nessun checkpoint valido')
        saved = read_json(cp)
        if saved['supervisor_state']['schema'] != Limits.PROFILE_SCHEMA:
            raise ValueError('Schema checkpoint non supportato')
        if any(saved['config'].get(k)!=v for k,v in asdict(c).items() if k!='generations'):
            raise ValueError('Resume configuration differs')
        if c.generations<saved['generation']:
            raise ValueError('Cannot resume backwards')
        state = saved['supervisor_state']
        if existing and (existing['generation']>state['generation'] or
                         existing['generation']==state['generation'] and existing['transaction']!=state['transaction']):
            raise ValueError('SMITH è avanzato in un’altra run: avviarne una nuova')
        save_checkpoint(directory/'latest.json', state)
        if saved['generation']==c.generations and (folder/'results.json').exists():
            return folder, saved['generation']
        population = [BooleanNet.restore(n) for n in saved['population']]
        scored = [(m, BooleanNet.restore(n)) for m,n in saved['evaluated_population']]
        history, start = saved['history'], saved['generation']+1
        reconcile_evolution_logs(folder, saved['generation'])
    else:
        if cp is not None or (folder/'config.json').exists():
            raise FileExistsError('Esperimento esistente: cambiare seed o riprendere il checkpoint')
        if existing:
            state = copy.deepcopy(existing)
            base = BooleanNet.restore(state['eva'])
            if (base.width>c.max_vector_width or base.memory_capacity!=c.memory_capacity or
                    base.internal_ticks!=c.internal_ticks or base.trace_depth!=c.trace_depth):
                raise ValueError('Budget larghezza/memoria incompatibili dal profilo SMITH: usare i valori salvati o un nuovo profilo')
            fit_topology(base, c.minimum_active_neurons, c.max_neurons, c.seed)
            state['eva'] = base.export()
        else:
            base = VectorBooleanNet(c.seed, neurons=c.initial_neurons, shape=shape,
                                    memory_capacity=c.memory_capacity, internal_ticks=c.internal_ticks,
                                    trace_depth=c.trace_depth)
            state = {'schema': Limits.PROFILE_SCHEMA, 'generation': 0, 'history': [],
                     'eva': base.export(), 'events': [], 'tasks': [], 'novelty': [],
                     'contributors': {}, 'promotions': 0, 'runs': [], 'transaction': 'initial',
                     'meta': {}, 'teacher_sources': [], 'explorers': []}
        history, start, population = [], 1, []
        for i in range(c.population):
            pool = state['explorers']
            net = BooleanNet.restore(pool[i]['brain'] if i<len(pool) else base.export())
            if net.width>c.max_vector_width:raise ValueError('Specialista oltre budget di larghezza')
            fit_topology(net, c.minimum_active_neurons, c.max_neurons, c.seed+i)
            if i>=len(pool) and i>0:
                net.recovery_mutate(c.seed+i*7919, c.max_neurons, allow_growth=False, allow_reactivate=False)
                net.vary_topology(c.seed+i*257, c.minimum_active_neurons, c.max_neurons)
            population.append(net)
        state['runs']=(state['runs']+[folder.name])[-Limits.RUN_NAMES_LIMIT:]
        atomic_json(folder/'config.json', asdict(c))
        atomic_json(folder/'manifest.json', {'schema': SCHEMA, 'engine': 'Predatore V6.05 Boolean Vector',
                    'experiment_id': folder.name, 'seed': c.seed, 'created_unix': time.time(),
                    'smith_profile': c.smith_profile, 'initial_revision': state['generation'],
                    'training_protocol': 'Independent adaptive explorers; teacher transfer; adaptive EVA; isolated validation/test copies'})
        atomic_json(folder/'brains/initial_eva.json', base.export())
        atomic_json(folder/'brains/initial_explorers.json', [net.export() for net in population])
    suites = prepare_suites(folder, c.size, c.seed)
    counts = {split: max(1, math.ceil(n/len(suites[split]))) for split,n in
              (('train', c.episodes), ('validation', c.validation_episodes), ('holdout', c.holdout_episodes))}
    atomic_json(folder/'environment/seed_protocol.json', {
        split: {s['name']: seeds_for(c.seed,split,s['name'],count) for s in suites[split]}
        for split,count in counts.items()})
    task = {'run': folder.name, 'suites': suites['validation'], 'seed': c.seed,
            'episodes': counts['validation'], 'steps': c.steps}
    state['tasks'] = [t for t in state['tasks'] if t['run']!=folder.name] + [task]
    if len(state['tasks'])>c.retention_tasks:
        state['tasks'] = ([state['tasks'][0]]+state['tasks'][-(c.retention_tasks-1):]
                          if c.retention_tasks>1 else [task])
    begin, cpu, last = time.perf_counter(), time.process_time(), start-1
    eva = BooleanNet.restore(state['eva'])
    eva.reset()
    cache = ProbeCache()
    for gen in range(start, c.generations+1):
        if should_stop and should_stop():
            break
        rng = random.Random(c.seed*1000003+gen)
        supervisor = state['meta']
        smith.choose(supervisor)
        strategy_context = state.get('pending_context', supervisor.get('last_context',0))
        strategy = state.get('pending_strategy', smith.choose(supervisor,strategy_context))
        train_suites,replay_archive,replayed=training_generation(c.size,c.seed,gen,counts['train'],
            state.get('training_replay',[]),folder.name)
        if monitor:monitor.training_suites=train_suites
        before_key=selection_key(probe_scores(eva,state['tasks'],cache),len(eva.nodes))
        incoming, scored, contributions = [], [], {}
        for slot, net in enumerate(population):
            source = f'{folder.name}/g{gen}/explorer{slot+1}'
            if observer:
                observer.set_context(generation=gen,individual=slot,phase='explorer',genome_id=net.snapshot_id())
            m, sample, seen = train_actor(net, source, train_suites, c, counts['train'], c.seed*99991+gen*257+slot, observer)
            incoming.extend(sample)
            contributions[source] = seen
            m['novelty'] = novelty(behavior(m), state['novelty'])
            scored.append((m,net))
        scored.sort(key=lambda pair: rank(pair[0]), reverse=True)
        teachers = choose_teachers(scored)
        current_sources = {m['source'] for m,_ in teachers}
        eligible_sources = set(state['teacher_sources']) | current_sources
        teaching_stream = list({e['event_id']:e for e in state['events']+incoming if e['source'] in eligible_sources}.values())
        distillation = {'current_teachers': sorted(current_sources), 'explorers_observed': len(scored)}
        proposals = build_transfer_proposals(eva, teachers, teaching_stream,
            contextual=c.contextual_distillation_enabled, focus_terminal=strategy=='consolidate',
            distillation_report=distillation)
        generation_start_id = eva.snapshot_id()
        neurons_before = len(eva.nodes)
        selected, improved, decisions = select_transfer(eva, proposals, state['tasks'], cache)
        promoted = selected is not None
        reason, consolidated = 'self_learning', 0
        if promoted:
            reason, eva, consolidated = selected
            eva.reset()
            state['promotions'] += 1
        distillation.update(contexts_consolidated=consolidated,
                            accepted=promoted and reason in ('contextual_distillation','teacher_distilled'))
        # EVA is a separate real learner, with the same train episodes/seeds as explorers.
        eva_before_training = eva.snapshot_id()
        pre_key=selection_key(probe_scores(eva,state['tasks'],cache),len(eva.nodes))
        if observer:
            observer.set_context(generation=gen,individual=c.population,phase='eva',genome_id=eva_before_training)
        best, sample, seen = train_actor(eva, f'{folder.name}/g{gen}/EVA', train_suites, c,
                                         counts['train'], c.seed*99991+gen*257+c.population, observer)
        incoming.extend(sample)
        after_key=selection_key(probe_scores(eva,state['tasks'],cache),len(eva.nodes))
        contributions[best['source']] = seen
        state['training_replay']=replay_archive
        state['transitions_total']=state.get('transitions_total',0)+sum(contributions.values())
        state['contributor_order']=(state.get('contributor_order',[])+list(contributions))[-Limits.CONTRIBUTOR_LIMIT:]
        state['contributors'].update(contributions)
        state['contributors']={k:state['contributors'][k] for k in state['contributor_order']}
        state['events'] = select_events(state['events'], incoming, c.experience_capacity, c.seed*100003+gen)
        present_sources = {e['source'] for e in state['events']}
        state['teacher_sources'] = sorted(eligible_sources & present_sources)
        state['novelty'] = (state['novelty']+[behavior(m) for m,_ in scored])[-c.novelty_archive_size:]
        comparison_by_scenario = {
            name: {'population_mean_capture': statistics.mean(m['by_scenario'][name]['capture_rate'] for m,_ in scored),
                   **population_completion([(m['by_scenario'][name],n) for m,n in scored])}
            for name in best['by_scenario']}
        summary = {'generation': gen, **best, **population_completion(scored),
            'agent': Limits.EVA_LABEL, 'measurement': 'adaptive_training',
            'population_mean_capture': statistics.mean(m['capture_rate'] for m,_ in scored),
            'population_mean_neurons': statistics.mean(len(n.nodes) for _,n in scored),
            'population_min_neurons': min(len(n.nodes) for _,n in scored),
            'population_max_neurons': max(len(n.nodes) for _,n in scored),
            'comparison_by_scenario': comparison_by_scenario,
            'comparison_protocol': 'Same maps, seeds, episode count and steps; lifetime learning enabled for EVA and every explorer.',
            'champion_id': eva.snapshot_id(), 'eva_revision': state['generation']+1,
            'eva_training_start_id': eva_before_training, 'eva_generation_start_id': generation_start_id,
            'eva_updated': eva.snapshot_id()!=generation_start_id,
            'eva_lifetime_changed': eva.snapshot_id()!=eva_before_training,
            'eva_promoted': promoted, 'promotion_reason': reason, 'strict_improvement': improved,
            'knowledge_entries': len(eva.knowledge), 'contextual_entries': len(eva.contextual_memory.rows),
            'eva_active_neurons': len(eva.nodes), 'minimum_active_neurons': c.minimum_active_neurons,
            'initial_neurons': c.initial_neurons, 'max_neurons': c.max_neurons,
            'topology_delta': len(eva.nodes)-neurons_before,
            'post_learning_regression':after_key<pre_key,'post_learning_improvement':after_key>pre_key,
            'replayed_training_episodes':replayed,'context_history':eva.context_codec.history,
            'associative_entries':len(eva.associative_memory.entries),
            'shared_events': len(state['events']), 'agent_contributors': len(scored),
            'local_circuit_candidates': sum(label.startswith('local_circuit_') for label,_,_ in proposals),
            'circuit_plasticity_enabled': c.circuit_plasticity_enabled,
            'transfer_candidates': len(proposals), 'transfer_accepted': promoted,
            'distillation': distillation, 'transitions_observed': sum(contributions.values()),
            'mean_novelty': statistics.mean(m['novelty'] for m,_ in scored),
            'archive_count': len(state['novelty']), 'genetic_diversity': len({n.snapshot_id() for _,n in scored})/len(scored),
            'plastic_updates': sum(n.update_count for _,n in scored)+eva.update_count,
            'smith_strategy': strategy}
        smith.observe(supervisor, strategy, after_key>before_key, context=strategy_context)
        next_context = smith.context_for(summary)
        next_strategy = smith.choose(supervisor,next_context)
        supervisor['last_context'] = next_context
        state.update(pending_context=next_context, pending_strategy=next_strategy)
        summary['smith_context'] = next_context
        history.append(summary)
        smith.record_run(supervisor, folder.name, history, state['generation']-gen+1)
        state['explorers'] = retain_explorers(scored)
        state['roles'] = {'population': 'explorers', 'eva': Limits.EVA_LABEL, 'smith': Limits.SMITH_LABEL}
        # Preserve trained explorers as elites; mutated descendants inherit personal memory.
        next_pop = [evaluation_copy(net) for _,net in scored[:c.elite]]
        novelty_order = sorted(scored, key=lambda p:p[0]['novelty'], reverse=True)
        while len(next_pop)<c.population:
            slot = len(next_pop)
            if slot%(2 if next_strategy=='explore' else 4)==0:
                parent = novelty_order[rng.randrange(max(1,len(novelty_order)//3))][1]
            elif slot%2==0:
                parent = eva
            else:
                parent = scored[rng.randrange(max(c.elite,c.population//3))][1]
            child = evaluation_copy(parent)
            if c.contextual_distillation_enabled:
                distill_into(child, teaching_stream, focus_terminal=next_strategy=='consolidate')
            mutation = child.recovery_mutate(rng.randrange(2**31),c.max_neurons,
                                             exploratory=next_strategy=='explore',allow_growth=False,allow_reactivate=False)
            mutation.update(child.vary_topology(rng.randrange(2**31),c.minimum_active_neurons,c.max_neurons,
                                                compact=next_strategy=='compact'))
            if next_strategy=='represent' or (gen+slot)%Limits.REPRESENTATION_MUTATION_PERIOD==0:
                mutation['representation']=child.mutate_representation(rng.randrange(2**31),c.max_vector_width)
            append_jsonl(folder/'evolution/mutations.jsonl',{'generation':gen+1,
                         'parent':parent.snapshot_id(),'child':child.snapshot_id(),**mutation})
            next_pop.append(child)
        population, last = next_pop, gen
        state['generation'] += 1
        state['eva'] = eva.export()
        state['transaction'] = f'{folder.name}:{gen}'
        state['config'] = {'profile': c.smith_profile, 'engine': Limits.ENGINE}
        state['history'] = (state['history']+[{'run':folder.name,**summary}])[-Limits.HISTORY_ENTRIES:]
        snapshot = {'schema': SCHEMA, 'generation': gen, 'config': asdict(c),
                    'population': [n.export() for n in population], 'history': history,
                    'evaluated_population': [(m,n.export()) for m,n in scored],
                    'supervisor_state': copy.deepcopy(state)}
        save_checkpoint(folder/'checkpoints/latest.json', snapshot)
        save_checkpoint(directory/'latest.json', state)
        atomic_json(directory/'status.json', {'profile':c.smith_profile,'revision':state['generation'],
            'run':folder.name,'generation':gen,'promotions':state['promotions'],
            'roles':state['roles'],'knowledge_entries':len(eva.knowledge),'contextual_entries':len(eva.contextual_memory.rows),
            'distillation':distillation,'smith_knowledge':smith.describe(supervisor),'shared_events':len(state['events']),
            'contributors':len(state['contributors']),'transitions':state['transitions_total'],
            'last_decisions':decisions,'transfer_accepted':promoted,'transfer_reason':reason,
            'retention_tasks':len(state['tasks']),'snapshot_id':eva.snapshot_id(),'meta':supervisor,
            'eva_neurons':len(eva.nodes),'eva_updated':summary['eva_updated']})
        atomic_json(folder/'brains/eva.json', eva.export())
        atomic_json(folder/f'environment/training/g{gen:06}.json',train_suites)
        circuits=folder/'neuroscience/generations'
        data=eva.export()
        keys=('engine','nodes','outputs','shape','rules','sensors','output_lanes')
        atomic_json(circuits/f'g{gen:06}.json',{**{k:data[k] for k in keys},'generation':gen})
        for old in sorted(circuits.glob('g*.json'))[:-Limits.CIRCUIT_HISTORY_LIMIT]:old.unlink()
        append_jsonl(folder/'evolution/generations.jsonl', summary)
        append_jsonl(folder/'evolution/consolidation.jsonl', {'generation':gen,'decisions':decisions,
            'transfer_selected':reason,'distillation':distillation,'contributors':contributions,
            'eva_lifetime_changed':summary['eva_lifetime_changed'],'topology_delta':summary['topology_delta']})
        if progress:
            progress(summary)
        if monitor:
            monitor.observe(summary, eva)
    if last==c.generations:
        _finish(folder,c,eva,suites,counts['validation'],counts['holdout'],begin,cpu,state,[net for _,net in scored])
    return folder,last


def _finish(folder,c,eva,suites,valid_count,test_count,begin,cpu,state,evaluated_population):
    atomic_json(folder/'brains/eva.json',eva.export())
    assessments = {split:assessment(eva,suites[split],c.seed,count,c.steps,split)
                   for split,count in (('validation',valid_count),('holdout',test_count))}
    atomic_json(folder/'benchmarks/results.json', assessments)
    initial=BooleanNet.restore(read_json(folder/'brains/initial_eva.json'))
    initial_test=assessment(initial,suites['holdout'],c.seed,test_count,c.steps,'holdout')
    atomic_json(folder/'benchmarks/learning_comparison.json',{'initial':initial_test,'final':assessments['holdout'],
        'capture_delta':assessments['holdout']['capture_rate']-initial_test['capture_rate']})
    population_tests = [assessment(net,suites['holdout'],c.seed,test_count,c.steps,'holdout') for net in evaluated_population]
    atomic_json(folder/'benchmarks/population_comparison.json', {
        'protocol':'Identical held-out maps/seeds/count/steps; adaptive isolated copies; no test feedback to agents or SMITH.',
        'generation':c.generations,'eva':assessments['holdout'],
        'population_mean_capture':statistics.mean(m['capture_rate'] for m in population_tests),
        'population_best_capture':max(m['capture_rate'] for m in population_tests),'individuals':population_tests})
    baseline = {name:baseline_suite(name,suites['holdout'],c.seed,'holdout',test_count,c.steps) for name in ('random','greedy')}
    atomic_json(folder/'benchmarks/baselines.json', baseline)
    for si,scenario in enumerate(suites['holdout']):
        for i,seed in enumerate(seeds_for(c.seed,'holdout',scenario['name'],min(3,test_count))):
            clone = evaluation_copy(eva)
            brain = clone.export()
            _,frames = rollout(clone,scenario,seed,c.steps,record=True)
            name = f'holdout_{i+1:02}.json' if si==0 else f'holdout_{scenario["name"]}_{i+1:02}.json'
            atomic_json(folder/'replays'/name,{'schema':SCHEMA,'seed':seed,'map':scenario['map_key'],
                'scenario':scenario['name'],'visibility':scenario['visibility'],'prey_moves':scenario['prey_moves'],
                'brain':brain,'frames':frames,'genome_id':eva.snapshot_id(),'learning':'isolated_adaptive'})
    atomic_json(folder/'neuroscience/brain_graph.json',eva.export())
    atomic_json(folder/'results.json',{'complete':True,'generations':c.generations,
        'engine':Limits.ENGINE,'elapsed_this_session_seconds':time.perf_counter()-begin,
        'cpu_this_session_seconds':time.process_time()-cpu,'champion_id':eva.snapshot_id(),
        'benchmarks':assessments,'baseline_comparison':baseline,'revision':state['generation'],
        'smith_profile':c.smith_profile,'note':'Adaptive isolated test; empirical selection, no guarantee of improvement.'})
    from .diagnostics import assess_champion
    assess_champion(eva,suites['holdout'],c.seed,min(4,test_count),c.steps,folder)
    seal(folder)
