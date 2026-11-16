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
            from engine.challenges import SCENARIOS
            options=SCENARIOS[scenario_name]
            map_path=exp/'environment'/'suites'/f'holdout_{scenario_name}.json'
            return {'map':read_json(map_path),'limit':limit,'seed':config['seed'],
                    'visibility':options['visibility'],'prey_moves':options['prey_moves'],
                    'scenario':scenario_name}
        map_path=exp/'environment'/'holdout.json'
        if not map_path.is_file():map_path=exp/'environment'/'train.json'
        return {'map':read_json(map_path),'limit':limit,'seed':config['seed'],
                'visibility':6,'prey_moves':False,'scenario':'static fallback'}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def episode_finished(actors, limit):
    return bool(actors) and all(a['world'].captured or a['world'].steps>=limit for a in actors)
