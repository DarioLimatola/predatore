"""Paired elapsed time of a worker alone and with the native Tk live arena open."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gui.laboratory import launch


def run_case(root,seed,live):
    command=[sys.executable,str(ROOT/'launcher.py'),'--train','--vault-root',str(root),
             '--seed',str(seed),'--population','4','--generations','4','--episodes','12',
             '--steps','32','--size','8',
             '--min-neurons','32','--base-neurons','32','--max-neurons','36','--vector-width','16',
             '--vector-shape','4,4','--memory-capacity','32','--experience-capacity','32']
    if live:command.append('--live')
    exp=root/'EvolutionVault/experiments'/f'EXP_seed{seed}_size8'
    row={'seed':seed,'live_window':live,'observed_frames':0}
    with (root/'worker.log').open('w') as log:
        def start():
            row['started']=time.perf_counter()
            return subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        if live:
            def observe(app,ui):
                ui['tabs'].select(ui['panels']['arena'])
                app.update_idletasks()
                process=start()
                last=None
                def follow():
                    nonlocal last
                    if exp.name!=ui['exp_name'].get() and (exp/'config.json').is_file():
                        ui['refresh_experiments'](prefer=exp.name)
                    current=ui['live_buffer'][0].current
                    if current:
                        packet,frame=current
                        sample=(packet['episode_id'],frame['t'])
                        if last!=sample:row['observed_frames']+=1;last=sample
                    if process.poll() is None:
                        app.after(15,follow)
                    else:
                        row['exit_code']=process.returncode
                        row['wall_s']=time.perf_counter()-row.pop('started')
                        ui['close_window']()
                app.after(15,follow)
            launch(root,test_hook=observe)
        else:
            process=start()
            row['exit_code']=process.wait()
            row['wall_s']=time.perf_counter()-row.pop('started')
    if row['exit_code']!=0:
        raise RuntimeError((root/'worker.log').read_text())
    if live and not row['observed_frames']:raise RuntimeError('No sampled frame displayed')
    result=json.loads((exp/'results.json').read_text())
    row['champion_id']=result['champion_id']
    row['holdout_captures']=result['benchmarks']['holdout']['capture_count']
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    differences=[]
    for i,seed in enumerate((701,702,703,704)):
        pair={}
        for enabled in ((True,False) if i%2 else (False,True)):
            with tempfile.TemporaryDirectory() as tmp:
                pair[enabled]=run_case(Path(tmp),seed,enabled)
        rows.extend(pair.values())
        a,b=pair[False],pair[True]
        if (a['champion_id'],a['holdout_captures'])!=(b['champion_id'],b['holdout_captures']):
            raise ValueError('Paired worker outputs differ')
        differences.append(100*(b['wall_s']/a['wall_s']-1))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report={'python':platform.python_version(),'platform':platform.platform(),'predeclared_threshold_pct':10,
            'protocol':'Identical CLI worker budget, fresh profile, 4 generations, 4 agents, 12 training episodes, 24 validation, 100 holdout, 32 steps. '
                       'Open Live Training Tk window before worker starts, render until worker exits. '
                       'Baseline worker with observer disabled. Alternating paired order.',
            'rows':rows,'paired_wall_delta_pct':differences,'median_paired_wall_delta_pct':statistics.median(differences)}
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
