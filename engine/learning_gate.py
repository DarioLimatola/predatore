"""Scenario-level evidence for decisions about a persistent EVA snapshot."""
from .paired_evidence import paired_capture_evidence


def scenario_evidence(tasks, baseline, candidate):
    """Comparable rows come from isolated copies with the same validation schedule."""
    rows = []
    for task, old, new in zip(tasks, baseline, candidate, strict=True):
        if old['by_scenario'].keys() != new['by_scenario'].keys():
            raise ValueError('Candidate and reference scenarios differ')
        for name in sorted(old['by_scenario']):
            a, b = old['by_scenario'][name], new['by_scenario'][name]
            if a['episode_count'] != b['episode_count']:
                raise ValueError('Candidate and reference episode budgets differ')
            rows.append({
                'task_run': task['run'], 'scenario': name, 'split': 'validation',
                'seed': task.get('seed'), 'episodes': a['episode_count'], 'steps': task.get('steps'),
                'capture_delta': b['capture_count'] - a['capture_count'],
                'stall_delta': b['stall_count'] - a['stall_count'],
                'reward_delta': b['mean_reward'] - a['mean_reward'],
                'paired_capture_evidence': paired_capture_evidence(a, b),
            })
    return rows


def exceeds_loss(rows, max_scenario_loss):
    return max_scenario_loss is not None and any(
        r['capture_delta'] < -max_scenario_loss for r in rows)


def proposal_channel(label):
    if label in ('contextual_distillation', 'teacher_distilled'):
        return 'distillation'
    if label.startswith('local_circuit_'):
        return 'structural'
    if label.startswith('teacher_'):
        return 'adoption'
    return 'personal'
