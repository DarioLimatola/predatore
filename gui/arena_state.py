"""GUI-independent Arena settings; safe when experiments move or disappear."""
from pathlib import Path
from engine.vault import read_json


def preview_settings(experiment, scenario_name=None):
    """Return cached settings once per preview load, never per Tkinter tick.

    Missing/incomplete experiments are ordinary UI states, not uncaught callbacks.
    """
    if not experiment:return None
    exp=Path(experiment)
    try:
        config=read_json(exp/'config.json')
        limit=int(config['steps'])
        if limit<1:raise ValueError('steps must be positive')
        if scenario_name and (exp/'environment'/'suites'/f'holdout_{scenario_name}.json').is_file():
            from engine.challenges import ALL_SCENARIOS
            options=ALL_SCENARIOS[scenario_name]
            map_path=exp/'environment'/'suites'/f'holdout_{scenario_name}.json'
            return {'map':read_json(map_path),'limit':limit,'seed':config['seed'],
                    'visibility':options['visibility'],'prey_moves':options['prey_moves'],
                    'name':scenario_name,'scenario':scenario_name}
        map_path=exp/'environment'/'holdout.json'
        if not map_path.is_file():map_path=exp/'environment'/'train.json'
        return {'map':read_json(map_path),'limit':limit,'seed':config['seed'],
                'visibility':6,'prey_moves':False,'scenario':'static fallback'}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def episode_finished(actors, limit):
    return bool(actors) and all(a['world'].captured or a['world'].steps>=limit for a in actors)


def frame_summary(frame):
    """Metrics from a sampled decision; t=0 carries position only."""
    from engine.world import ACTIONS
    action = frame.get('action')
    action_name = ACTIONS[action] if type(action) is int and 0 <= action < len(ACTIONS) else '—'
    fields = [f'passo {frame.get("t", "—")}', f'azione {action_name}']
    obs = frame.get('observation')
    if obs:
        fields.append('preda visibile' if obs[0] else 'preda fuori vista')
    source = frame.get('decoder', {}).get('source')
    if source:
        fields.append(f'decoder {source}')
    if 'captured' in frame:
        fields.append('catturata' if frame['captured'] else 'in corso')
    if 'knowledge_entries' in frame:
        fields.append(f'RAM {frame["knowledge_entries"]} voci')
    events = frame.get('world_events') or {}
    if events.get('due_tick') is not None:
        fields.append(f'contatto {events["tag_tick"]}, conferma prevista {events["due_tick"]}')
    prey_action = events.get('prey_action')
    if type(prey_action) is int and 0 <= prey_action < len(ACTIONS):
        fields.append(f'azione preda {ACTIONS[prey_action]}')
    return ' · '.join(fields)
