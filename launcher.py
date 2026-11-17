"""CLI and optional Tkinter viewer. Run: python launcher.py --help"""
import argparse
from pathlib import Path
from engine.evolution import Config,run
from engine.settings import Limits

def main():
    p=argparse.ArgumentParser(description='Predatore V6.05 — EVA learning agent / SMITH agent supervisor')
    p.add_argument('--monitor-every',type=int,default=0,help='Live validation every N generations; 0 disables extra monitoring')
    p.add_argument('--live',action='store_true',help='Publish sampled real training frames')
    p.add_argument('--challenge-tests',metavar='EXPERIMENT_DIR',help='Independent invisible-prey and required-pounce probes')
    p.add_argument('--diagnose',metavar='EXPERIMENT_DIR',help='Paired behavioral and neuron ablations, isolated output')
    p.add_argument('--diagnostic-output',default='Analysis',help='Output directory for standalone diagnostics')
    p.add_argument('--train',action='store_true',help='Run evolutionary training')
    p.add_argument('--resume',action='store_true',help='Resume a saved experiment')
    p.add_argument('--smith-profile',default=Limits.DEFAULT_SMITH_PROFILE,help='Persistent SMITH memory and strategy identity')
    p.add_argument('--max-vector-width',type=int,default=Limits.MAX_EVOLVED_WIDTH)
    p.add_argument('--vector-width',type=int,default=Limits.DEFAULT_WIDTH)
    p.add_argument('--vector-shape',default='',help='Module axes, e.g. 4,4,4 for 64 bits')
    p.add_argument('--base-neurons',type=int,default=Limits.DEFAULT_BASE_NEURONS)
    p.add_argument('--min-neurons',type=int,default=Limits.DEFAULT_MIN_NEURONS)
    p.add_argument('--max-neurons',type=int,default=Limits.DEFAULT_MAX_NEURONS)
    p.add_argument('--memory-capacity',type=int,default=Limits.DEFAULT_MEMORY)
    p.add_argument('--experience-capacity',type=int,default=Limits.DEFAULT_EXPERIENCES)
    p.add_argument('--internal-ticks',type=int,default=Limits.DEFAULT_TICKS)
    p.add_argument('--trace-depth',type=int,default=Limits.DEFAULT_TRACE)
    p.add_argument('--no-circuit-plasticity',action='store_true',default=None,
                   help='Disable local circuit candidate learning (for controlled ablation)')
    p.add_argument('--no-context-distillation',action='store_true',default=None,help='Controlled ablation: disable contextual consolidation')
    p.add_argument('--retention-tasks',type=int,default=Limits.DEFAULT_RETENTION_TASKS)
    p.add_argument('--smith-status',action='store_true')
    p.add_argument('--generations',type=int,default=Limits.DEFAULT_GENERATIONS)
    p.add_argument('--population',type=int,default=Limits.DEFAULT_POPULATION)
    p.add_argument('--episodes',type=int,default=Limits.DEFAULT_EPISODES)
    p.add_argument('--steps',type=int,default=Limits.DEFAULT_STEPS)
    p.add_argument('--size',type=int,default=Limits.DEFAULT_WORLD_SIZE)
    p.add_argument('--seed',type=int,default=Limits.DEFAULT_SEED)
    p.add_argument('--vault-root',default='.')
    p.add_argument('--verify',metavar='EXPERIMENT_DIR')
    p.add_argument('--gui',action='store_true',help='Open unified V6 laboratory')
    args=p.parse_args()
    if args.smith_status:
        from engine.continual import load_profile
        import json
        state=load_profile(args.vault_root,args.smith_profile)
        print(json.dumps({'profile':args.smith_profile,'revision':state['generation'] if state else 0,
              'runs':state['runs'] if state else [],'knowledge':len(state['eva'].get('knowledge',{}).get('rows',[])) if state else 0,
              'contextual_memory':len(state['eva'].get('contextual_memory',{}).get('rows',[])) if state else 0,
              'explorers':len(state.get('explorers',[])) if state else 0,
              'smith_runs':len(state.get('meta',{}).get('run_memory',[])) if state else 0},indent=2))
        return
    if args.challenge_tests:
        from engine.supplementary import run_probes
        out,_=run_probes(args.challenge_tests,args.diagnostic_output)
        print('Prove supplementari:',out)
        return
    if args.diagnose:
        from engine.boolean_net import BooleanNet
        from engine.vault import read_json
        from engine.diagnostics import assess_champion
        import time
        exp=Path(args.diagnose)
        cfg=read_json(exp/'config.json')
        from engine.challenges import load_suites
        suites=load_suites(exp,'holdout')
        out=Path(args.diagnostic_output)/f'{exp.name}_{time.time_ns()}'
        assess_champion(BooleanNet.restore(read_json(exp/'brains/eva.json')),suites,cfg['seed'],
                        4,cfg['steps'],out,neuron_ablations=True)
        print('Diagnostics:',out)
        return
    if args.verify:
        from engine.vault import verify
        outcome=verify(args.verify)
        for name,ok in outcome.items():print(('OK' if ok else 'FAIL'),name)
        raise SystemExit(0 if outcome and all(outcome.values()) else 1)
    if args.train:
        from dataclasses import fields
        from engine.vault import latest_checkpoint_path, read_json
        folder=Path(args.vault_root)/'EvolutionVault'/'experiments'/f'EXP_seed{args.seed}_size{args.size}'
        if args.resume:
            found=latest_checkpoint_path(folder/'checkpoints')
            if found is None:raise FileNotFoundError('No valid checkpoint to resume')
            saved=read_json(found)['config']
            cfg=Config(**{k:v for k,v in saved.items() if k in {f.name for f in fields(Config)}})
            cfg.generations=args.generations
            if args.no_context_distillation is not None and cfg.contextual_distillation_enabled:
                raise ValueError('Cannot change contextual distillation during resume')
            if args.no_circuit_plasticity is not None:
                requested=not args.no_circuit_plasticity
                if requested!=cfg.circuit_plasticity_enabled:
                    raise ValueError('Cannot change circuit plasticity during resume')
        else:
            cfg=Config(generations=args.generations,population=args.population,episodes=args.episodes,
                       steps=args.steps,size=args.size,seed=args.seed,smith_profile=args.smith_profile,vector_width=args.vector_width,max_vector_width=args.max_vector_width,
                       memory_capacity=args.memory_capacity,experience_capacity=args.experience_capacity,
                       vector_shape=args.vector_shape,max_neurons=args.max_neurons,
                       initial_neurons=args.base_neurons,minimum_active_neurons=args.min_neurons,
                       internal_ticks=args.internal_ticks,trace_depth=args.trace_depth,
                       circuit_plasticity_enabled=not args.no_circuit_plasticity,
                       contextual_distillation_enabled=not args.no_context_distillation,
                       retention_tasks=args.retention_tasks)
        stop_file=folder/'stop_requested'
        if stop_file.exists():stop_file.unlink()
        observer=None
        if args.live:
            from engine.telemetry import LiveObserver
            observer=LiveObserver(folder)
        monitor=None
        if args.monitor_every>0:
            from engine.progress_monitor import ProgressMonitor
            monitor=ProgressMonitor(folder,cfg,args.monitor_every)
        status='failed'
        try:
            folder,g=run(args.vault_root,cfg,resume=args.resume,should_stop=lambda:stop_file.exists(),observer=observer,monitor=monitor,
                         progress=lambda x:print(f"Gen {x['generation']:4d} | EVA - learning agent {x['capture_rate']:.1%} | avg {x['population_mean_capture']:.1%} | neurons {x['active_neurons']}",flush=True))
            status='finished' if g==cfg.generations else 'stopped'
        finally:
            if observer:observer.close(status)
        print('Experiment:',folder,'completed generation:',g)
    if args.gui or not args.train and not args.verify:
        from gui.laboratory import launch
        launch(Path(args.vault_root).resolve())
if __name__=='__main__':main()
