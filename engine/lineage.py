"""Fork a historical experiment and the exact SMITH transaction it committed."""
import os
from pathlib import Path
import shutil
import tempfile

from .continual import profile_path
from .settings import Limits
from .vault import (LEGACY_SCHEMA, SCHEMA, atomic_json, fingerprint, latest_checkpoint_path,
                    read_json, experiment_input_hashes, reconcile_evolution_logs, save_checkpoint, seal, verify)


def fork_v605(source, new_root):
    source, destination = Path(source).resolve(), Path(new_root).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('La destinazione deve essere esterna alla run originale.')
    if destination.exists():
        raise FileExistsError(f'Scegliere una nuova cartella per la linea sperimentale: {destination}')
    original = latest_checkpoint_path(source / 'checkpoints')
    if original is None:
        raise ValueError('Checkpoint V6.05 mancante o illeggibile.')
    snapshot = read_json(original)
    if snapshot.get('schema') != LEGACY_SCHEMA or snapshot.get('supervisor_state', {}).get('schema') != Limits.PROFILE_SCHEMA:
        raise ValueError('La migrazione richiede un checkpoint di esperimento V6.05.')
    config = snapshot['config']
    run_name = f"EXP_seed{config['seed']}_size{config['size']}"
    if snapshot['supervisor_state'].get('transaction') != f"{run_name}:{snapshot['generation']}":
        raise ValueError('La transazione SMITH non corrisponde alla run richiesta.')
    source_digest = fingerprint(original)
    if (source / 'integrity/checksums.json').is_file() and not all(verify(source).values()):
        raise ValueError('Verifica SHA-256 della run V6.05 fallita.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.predatore-fork-') as stage:
        root = Path(stage)
        experiment = root / 'EvolutionVault' / 'experiments' / run_name
        shutil.copytree(source, experiment, ignore=shutil.ignore_patterns(
            'live', 'gui_worker.log', 'stop_requested', 'writer.lock', '*.tmp'))
        copied = experiment / 'checkpoints' / original.name
        if fingerprint(copied) != source_digest or not (snapshot == read_json(copied)):
            raise ValueError('Il checkpoint è cambiato durante la copia. Riprovare a training fermo.')
        if (experiment / 'integrity/checksums.json').is_file() and not all(verify(experiment).values()):
            raise ValueError('La copia della run non supera la verifica SHA-256.')
        origin = {'schema_version': LEGACY_SCHEMA, 'engine_version': '6.05',
                  'checkpoint': original.name, 'checkpoint_sha256': source_digest,
                  'generation': snapshot['generation'], 'transaction': snapshot['supervisor_state']['transaction']}
        # Keep the historical codecs and genomes. Only the new lineage may advance.
        shutil.rmtree(experiment / 'checkpoints')
        state = snapshot['supervisor_state']
        state['engine_version'] = Limits.VERSION
        snapshot.update(schema=SCHEMA, schema_version=SCHEMA, engine_version=Limits.VERSION, origin=origin)
        save_checkpoint(profile_path(root, config['smith_profile']) / 'latest.json', state)
        manifest = read_json(experiment / 'manifest.json')
        manifest.update(schema=SCHEMA, schema_version=SCHEMA, engine_version=Limits.VERSION,
                        engine=f'Predatore V{Limits.VERSION} Boolean Vector', origin=origin,
                        genome_codec=state['eva'].get('engine', Limits.ENGINE), checkpoint_codec='canonical-json-utf8/sha256-v1')
        atomic_json(experiment / 'manifest.json', manifest)
        snapshot['input_sha256'] = experiment_input_hashes(experiment)
        save_checkpoint(experiment / 'checkpoints/latest.json', snapshot)
        reconcile_evolution_logs(experiment, snapshot['generation'])
        seal(experiment)
        # mkdir claims a new destination exclusively, including simultaneous forks.
        destination.mkdir()
        try:
            os.rename(root / 'EvolutionVault', destination / 'EvolutionVault')
        except OSError:
            destination.rmdir()
            raise
    return destination / 'EvolutionVault' / 'experiments' / run_name
