"""Predatore V6 laboratory UI.

Tkinter remains on the main thread. Training happens in a separate Python process.
Live Training observes real worker frames. Independent Preview simulates saved genotypes.
Recorded Replay uses actual logged positions and actions.
"""
import colorsys
import json
import math
import random
import re
import subprocess
from types import SimpleNamespace
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from engine.boolean_net import BooleanNet
from engine.settings import Limits
from engine.evolution import Config
from engine.world import DIRS
from engine.challenge_worlds import make_world
from engine.challenges import ALL_SCENARIOS
from engine.vault import read_json, latest_checkpoint_path, checkpoint_paths
from gui.arena_state import preview_settings, episode_finished, frame_summary
from engine.telemetry import LiveBuffer, write_mailbox
from gui.brain_inspector import BrainInspector
from gui.progress_chart import EVA_COLOR, POPULATION_COLOR

BG = '#edf2f8'
PANEL = '#ffffff'
INK = '#14263e'
MUTED = '#64748b'
ACCENT = '#2563eb'
COLORS = ['#2563eb', '#16a34a', '#8b5cf6', '#d97706', '#db2777', '#0891b2', '#a16207', '#64748b']


SCENARIO_LABELS={'static':'Preda ferma','moving':'Preda mobile','maze':'Ostacoli classici','hidden':'Visibilità ridotta (3 celle)',
                 'maze_novel':'Labirinto a corridoi','delayed_reward':'Ricompensa ritardata',
                 'intelligent_prey':'Preda neurale'}
SCENARIO_IDS={value:key for key,value in SCENARIO_LABELS.items()}
def scenario_id(value):return SCENARIO_IDS.get(value,value)
def agent_color(index):
    rgb=colorsys.hsv_to_rgb((index*.61803398875)%1,.72,.72)
    return '#'+''.join(f'{round(v*255):02x}' for v in rgb)
def friendly(value):
    return {'train':'Addestramento','validation':'Validazione','holdout':'Test finale',
            'random':'Movimenti casuali (4 direzioni)','greedy':'Inseguitore privilegiato','sensor_greedy':'Inseguitore con sensori',**SCENARIO_LABELS}.get(value,value)

def available_agent_specs(exp, comparison="greedy"):
    """The actual EVA and explorers evaluated in this generation."""
    exp = Path(exp)
    specs = []
    file = exp/'brains/eva.json'
    if file.is_file():
        try:
            specs.append({'name':Limits.EVA_LABEL,'color':EVA_COLOR,
                          'brain':BooleanNet.restore(read_json(file))})
        except (OSError,ValueError,KeyError,TypeError,IndexError):
            pass
    checkpoint = latest_checkpoint_path(exp / 'checkpoints')
    if checkpoint is not None:
        try:
            cp = read_json(checkpoint)
            for i, (_,item) in enumerate(cp.get('evaluated_population', [])):
                specs.append({'name': f'Esploratore {i+1} — Gen {cp.get("generation", "?")}',
                              'color': agent_color(i+2), 'brain': BooleanNet.restore(item)})
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    specs.append({'name': friendly(comparison), 'color': '#d97706', 'baseline': comparison})
    specs.append({'name': 'Movimenti casuali (4 direzioni)', 'color': '#64748b', 'baseline': 'random'})
    return specs


def preview_action(actor):
    world = actor['world']
    if actor.get('baseline') == 'random':
        return actor['rng'].randrange(4)
    if actor.get('baseline') == 'sensor_greedy':
        from engine.evaluation import sensor_greedy
        return sensor_greedy(world.observe(),actor['rng'])
    if actor.get('baseline') == 'greedy':
        # Diagnostic heuristic; accesses true prey coordinates, unlike the EVA sensors.
        options = []
        for index, (dx, dy) in enumerate(DIRS):
            q = (world.predator[0] + dx, world.predator[1] + dy)
            if world.free(q):
                options.append((abs(q[0]-world.prey[0]) + abs(q[1]-world.prey[1]), index))
        return min(options)[1] if options else 0
    return actor['brain'].decide(world.observe())


def launch(root_dir=None, *, test_hook=None):
    root_dir = Path(root_dir or Path(__file__).resolve().parents[1]).resolve()
    vault = root_dir / 'EvolutionVault' / 'experiments'
    vault.mkdir(parents=True, exist_ok=True)

    app = tk.Tk()
    app.title(f'Predatore {Limits.VERSION} · Laboratorio evolutivo')
    app.geometry('1240x830')
    app.minsize(950, 640)
    app.configure(background=BG)
    style = ttk.Style(app)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background=BG, foreground=INK)
    style.configure('TFrame', background=BG)
    style.configure('Panel.TFrame', background=PANEL)
    style.configure('TLabel', background=BG, foreground=INK)
    style.configure('Muted.TLabel', background=BG, foreground=MUTED)
    style.configure('Panel.TLabel', background=PANEL, foreground=INK)
    style.configure('TButton', padding=(11, 7))
    style.configure('Primary.TButton', background=ACCENT, foreground='white', padding=(13, 9), font=('Segoe UI', 10, 'bold'))
    style.map('Primary.TButton', background=[('active', '#1d4ed8')])
    style.configure('TNotebook', background=BG, borderwidth=0)
    style.configure('TNotebook.Tab', padding=(20, 11), background='#dbe5f1', foreground=INK)
    style.map('TNotebook.Tab', background=[('selected', PANEL)], foreground=[('selected', ACCENT)])
    style.configure('TLabelFrame', background=BG, foreground=INK)
    style.configure('TLabelFrame.Label', background=BG, foreground=INK, font=('Segoe UI', 10, 'bold'))
    style.configure('Treeview', rowheight=30, background=PANEL, fieldbackground=PANEL)
    style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))

    header = tk.Frame(app, bg='#14263e', height=74)
    header.pack(fill='x')
    header.pack_propagate(False)
    tk.Label(header, text='PREDATORE', font=('Segoe UI', 19, 'bold'), bg='#14263e', fg='white').pack(side='left', padx=(22, 7), pady=12)
    tk.Label(header, text=f'{Limits.VERSION}   /   Laboratorio evolutivo', font=('Segoe UI', 10, 'bold'), bg='#14263e', fg='#93c5fd').pack(side='left', pady=16)
    header_state = tk.StringVar(value='Nessun addestramento in corso')
    tk.Label(header, textvariable=header_state, font=('Segoe UI', 10), bg='#14263e', fg='#cbd5e1').pack(side='right', padx=22)

    tabs = ttk.Notebook(app)
    tabs.pack(fill='both', expand=True, padx=14, pady=(12, 8))
    panels = {}
    for key, name in [('training','Esperimento'), ('arena','Arena e replay'), ('results','Risultati'),
                      ('brain','Cervello e memoria'), ('archive','Archivio')]:
        panels[key] = ttk.Frame(tabs)
        tabs.add(panels[key], text=name)
    notebooks = {}
    for group, views in [('results', [('evolution','Evoluzione'), ('benchmark','Benchmark')]),
                         ('archive', [('supervisor','SMITH e checkpoint'), ('diagnostics','Diagnostica')])]:
        notebook = ttk.Notebook(panels[group])
        notebook.pack(fill='both', expand=True)
        notebooks[group] = notebook
        for key, name in views:
            panels[key] = ttk.Frame(notebook)
            notebook.add(panels[key], text=name)
    footer = tk.Label(app, text=f'Predatore {Limits.VERSION}   ·   EvolutionVault',
                      bg=BG, fg=MUTED, font=('Segoe UI', 9))
    footer.pack(anchor='w', padx=22, pady=(0, 8))

    # Shared experiment selector, visible on all non-training panels.
    chooser = ttk.Frame(app)
    chooser.pack(before=tabs, fill='x')
    chooser.pack(fill='x', padx=15, pady=(12, 5))
    ttk.Label(chooser, text='ESPERIMENTO', font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 8))
    exp_name = tk.StringVar()
    exp_combo = ttk.Combobox(chooser, textvariable=exp_name, state='readonly', width=36)
    exp_combo.pack(side='left', padx=4)
    ttk.Label(chooser, text='Esperimenti salvati in EvolutionVault',
              style='Muted.TLabel').pack(side='right', padx=7)
    def active_experiment():
        return vault / exp_name.get() if exp_name.get() else None

    # TRAINING: keep experiment controls together with the live laboratory.
    cfg_box = ttk.LabelFrame(panels['training'], text='Configurazione del prossimo esperimento')
    cfg_box.pack(fill='x', padx=16, pady=(14, 8))
    default_config=Config()
    defaults = {key:getattr(default_config,key) for key in ('population','generations','episodes','steps','size','seed')}
    fields = {}
    for idx, (key, val) in enumerate(defaults.items()):
        col, row = idx % 3, idx // 3
        slot = ttk.Frame(cfg_box)
        slot.grid(row=row, column=col, sticky='ew', padx=14, pady=8)
        ttk.Label(slot, text={'population':'Esploratori','generations':'Generazioni','episodes':'Episodi','steps':'Passi massimi','size':'Lato della mappa','seed':'Seed'}[key], style='Muted.TLabel', font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        v = tk.StringVar(value=str(val))
        fields[key] = v
        ttk.Entry(slot, textvariable=v, width=20).pack(anchor='w', pady=(4, 0))
    for i in range(3): cfg_box.columnconfigure(i, weight=1)
    smith_profile=tk.StringVar(value=Limits.DEFAULT_SMITH_PROFILE)
    profile_row=ttk.Frame(cfg_box)
    profile_row.grid(row=2,column=0,columnspan=3,sticky='w',padx=14,pady=6)
    ttk.Label(profile_row,text='Profilo SMITH').pack(side='left')
    ttk.Entry(profile_row,textvariable=smith_profile,width=22).pack(side='left',padx=8)
    ttk.Label(profile_row,text='Stesso SMITH = archivio, strategie ed EVA continuano tra le run.').pack(side='left')
    vector_max=tk.StringVar(value=str(Limits.MAX_EVOLVED_WIDTH))
    vector_shape=tk.StringVar(value=','.join(map(str,Limits.DEFAULT_SHAPE)))
    memory_budget=tk.StringVar(value=str(Limits.DEFAULT_MEMORY))
    experience_budget=tk.StringVar(value=str(Limits.DEFAULT_EXPERIENCES))
    vector_row=ttk.Frame(cfg_box)
    vector_row.grid(row=3,column=0,columnspan=3,sticky='w',padx=14,pady=5)
    for label,var in [('Forma iniziale',vector_shape),('Memorie RAM',memory_budget),('Esperienze condivise',experience_budget),('Max bit/modulo',vector_max)]:
        ttk.Label(vector_row,text=label).pack(side='left',padx=(0,5))
        ttk.Entry(vector_row,textvariable=var,width=10).pack(side='left',padx=(0,14))
    structure_row=ttk.Frame(cfg_box)
    structure_row.grid(row=4,column=0,columnspan=3,sticky='ew',padx=14,pady=5)
    neuron_controls={}
    for column,(key,label,default) in enumerate((
            ('minimum','Minimo',Limits.DEFAULT_MIN_NEURONS),
            ('base','Base',Limits.DEFAULT_BASE_NEURONS),
            ('maximum','Massimo',Limits.DEFAULT_MAX_NEURONS))):
        frame=ttk.Frame(structure_row)
        frame.grid(row=0,column=column,sticky='ew',padx=(0,18))
        enabled=tk.BooleanVar(value=False)
        value=tk.IntVar(value=default)
        scale=ttk.Scale(frame,from_=Limits.MIN_NEURONS,to=Limits.MAX_NEURONS,
                        variable=value,orient='horizontal',length=170,state='disabled',
                        command=lambda v,var=value:var.set(round(float(v))))
        spin=ttk.Spinbox(frame,from_=Limits.MIN_NEURONS,to=Limits.MAX_NEURONS,
                         textvariable=value,width=6,state='disabled')
        def toggle(enabled=enabled,scale=scale,spin=spin,value=value,default=default):
            state='normal' if enabled.get() else 'disabled'
            if not enabled.get():value.set(default)
            scale.configure(state=state);spin.configure(state=state)
        ttk.Checkbutton(frame,text=label+' personalizzato',variable=enabled,
                        command=toggle).pack(anchor='w')
        scale.pack(side='left');spin.pack(side='left',padx=4)
        neuron_controls[key]={'enabled':enabled,'value':value,'scale':scale,'spin':spin,
                              'toggle':toggle,'default':default}
    ttk.Label(cfg_box,text='Predefiniti: min 32 · base 64 · max 128. La base inizializza nuove reti; le strutture vincenti si ereditano.',
              style='Muted.TLabel').grid(row=5,column=0,columnspan=3,sticky='w',padx=14,pady=5)
    experiments = ttk.Frame(cfg_box)
    experiments.grid(row=6,column=0,columnspan=3,sticky='w',padx=14,pady=6)
    scenario_set=tk.StringVar(value='Classici')
    ttk.Label(experiments,text='Scenari').pack(side='left')
    ttk.Combobox(experiments,textvariable=scenario_set,values=('Classici','Estesi'),state='readonly',
                 width=8).pack(side='left',padx=(4,12))
    relation_horizon=tk.StringVar(value='0')
    ttk.Label(experiments,text='Traccia relazioni (passi)').pack(side='left')
    ttk.Spinbox(experiments,from_=0,to=64,width=4,textvariable=relation_horizon).pack(side='left',padx=(4,12))
    scenario_loss=tk.StringVar(value='')
    ttk.Label(experiments,text='Perdite ammesse/scenario¹').pack(side='left')
    ttk.Entry(experiments,textvariable=scenario_loss,width=4).pack(side='left',padx=(4,12))
    select_for_learning=tk.BooleanVar(value=False)
    ttk.Checkbutton(experiments,text='Seleziona capacità di apprendere',variable=select_for_learning).pack(side='left')
    ttk.Label(cfg_box,text='¹ Vuoto disattiva la soglia. Una traccia di 0 passi disattiva il credito relazionale.',
              style='Muted.TLabel').grid(row=7,column=0,columnspan=3,sticky='w',padx=14,pady=(0,4))
    eva_status=tk.StringVar(value=Limits.EVA_LABEL+': in attesa della prima run')
    ttk.Label(panels['training'],textvariable=eva_status,wraplength=1100).pack(anchor='w',padx=18,pady=5)
    state_cache=[None,None]
    def selected_state():
        # The displayed run controls these panels, independently of next-run inputs.
        exp=active_experiment()
        if exp:
            cp=latest_checkpoint_path(exp/'checkpoints')
            if cp is None:return None,None
            stamp=(str(cp),cp.stat().st_mtime_ns,cp.stat().st_size)
            if stamp!=state_cache[0]:
                data=read_json(cp)
                state_cache[:]=[stamp,(data['config']['smith_profile'],data['supervisor_state'])]
            return state_cache[1]
        from engine.continual import load_profile
        return smith_profile.get(),load_profile(root_dir,smith_profile.get())
    def refresh_eva():
        try:
            profile,state=selected_state()
            if state is None:raise ValueError('No state')
            brain=state['eva']
            eva_status.set(f"{Limits.EVA_LABEL} · revisione {state['generation']} · neuroni {len(brain['nodes'])} · RAM {len(brain['knowledge']['rows'])} · contesti {len(brain['contextual_memory']['rows'])} | SMITH {profile} · esperienze {len(state['events'])} · trasferimenti {state['promotions']}")
        except (OSError,ValueError,KeyError,TypeError):
            eva_status.set(Limits.EVA_LABEL+': nessun dato per la run selezionata. SMITH conserva memoria e strategie.')
    smith_profile.trace_add('write',lambda *_:refresh_eva())
    tbar = ttk.Frame(panels['training'])
    tbar.pack(fill='x', padx=16, pady=8)
    train_status = tk.StringVar(value='Imposta i parametri e avvia una nuova run.')
    ttk.Label(panels['training'], textvariable=train_status, font=('Segoe UI', 11, 'bold'), wraplength=1020).pack(anchor='w', padx=18, pady=(2, 8))
    logs = tk.Text(panels['training'], height=12, wrap='word', bg='#17243b', fg='#e2e8f0',
                   insertbackground='white', borderwidth=0, padx=13, pady=12, font=('Consolas', 10))
    logs.pack(fill='both', expand=True, padx=16, pady=(6, 16))
    worker = [None]
    worker_experiment = [None]
    worker_log = [None]
    loaded_checkpoint = [None]
    displayed_log = [None]
    def input_config():
        try: c = {k:int(v.get()) for k,v in fields.items()}
        except ValueError: raise ValueError('All settings must be integers')
        if not (Limits.MIN_POPULATION<=c['population']<=Limits.MAX_POPULATION and 1<=c['generations']<=Limits.MAX_GENERATIONS and 1<=c['episodes']<=Limits.MAX_EPISODES
                and 1<=c['steps']<=Limits.MAX_STEPS and Limits.MIN_WORLD_SIZE<=c['size']<=Limits.MAX_WORLD_SIZE):
            raise ValueError('Settings out of range')
        return c
    def start_training(resume=False):
        if worker[0] and worker[0].poll() is None:
            messagebox.showinfo('Training', 'An experiment is already running.');return
        try: c = input_config()
        except ValueError as exc: messagebox.showerror('Configuration', str(exc));return
        folder = vault / f'EXP_seed{c["seed"]}_size{c["size"]}'
        if not resume and (checkpoint_paths(folder/'checkpoints') or (folder/'config.json').exists()):
            messagebox.showwarning('Experiment already exists', 'Choose another seed or Resume checkpoint. Existing experiments are not overwritten.')
            return
        if resume and latest_checkpoint_path(folder/'checkpoints') is None:
            messagebox.showwarning('No checkpoint', 'Start a new experiment first.');return
        if resume:
            try:
                saved=read_json(latest_checkpoint_path(folder/'checkpoints'))
                if saved['supervisor_state']['schema']!=Limits.PROFILE_SCHEMA:
                    raise ValueError('Schema checkpoint non supportato.')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                messagebox.showerror('Cannot resume',str(exc));return
        folder.mkdir(parents=True, exist_ok=True)
        flag = folder/'stop_requested'
        if flag.exists():flag.unlink()
        cmd = [sys.executable, '-u', str(Path(__file__).resolve().parents[1]/'launcher.py'), '--train', '--vault-root', str(root_dir),
               *[f'--{k}={v}' for k,v in c.items()]]
        cmd.extend(['--live','--monitor-every','10','--smith-profile',smith_profile.get()])
        if not resume:
            from engine.continual import load_profile
            try:
                horizon=int(relation_horizon.get())
                if not 0 <= horizon <= 64:raise ValueError('Traccia relazioni: scegliere 0–64 passi.')
                cmd.extend(['--scenario-set','extended' if scenario_set.get()=='Estesi' else 'classic',
                            '--relation-credit-horizon',str(horizon)])
                if scenario_loss.get().strip():
                    loss=int(scenario_loss.get())
                    if loss < 0:raise ValueError('Le perdite ammesse devono essere un intero positivo o zero.')
                    cmd.extend(['--max-scenario-loss',str(loss)])
                if select_for_learning.get():cmd.append('--select-for-learning')
                bounds={key:int(control['value'].get()) if control['enabled'].get() else control['default']
                        for key,control in neuron_controls.items()}
                if not Limits.MIN_NEURONS<=bounds['minimum']<=bounds['base']<=bounds['maximum']<=Limits.MAX_NEURONS:
                    raise ValueError('Neuroni: 32 <= minimo <= base <= massimo <= 4096')
                existing=load_profile(root_dir,smith_profile.get())
                if existing:
                    brain=existing['eva']
                    cmd.extend(['--vector-width',str(math.prod(brain['shape'])),'--vector-shape',','.join(map(str,brain['shape'])),'--memory-capacity',str(brain['memory_capacity']),'--internal-ticks',str(brain['internal_ticks']),'--trace-depth',str(brain['trace_depth'])])
                else:
                    shape=tuple(int(x) for x in vector_shape.get().split(','))
                    cmd.extend(['--max-vector-width',str(int(vector_max.get())),'--vector-width',str(math.prod(shape)),'--vector-shape',vector_shape.get(),'--memory-capacity',str(int(memory_budget.get()))])
                cmd.extend(['--experience-capacity',str(int(experience_budget.get())),
                            '--min-neurons',str(bounds['minimum']),'--base-neurons',str(bounds['base']),
                            '--max-neurons',str(bounds['maximum'])])
            except (OSError,ValueError,KeyError) as exc:
                messagebox.showerror('Profilo SMITH',str(exc));return
        if resume:cmd.append('--resume')
        logpath = folder / 'gui_worker.log'
        try:
            stream = logpath.open('a', encoding='utf-8')
            flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) if sys.platform=='win32' else 0
            worker[0] = subprocess.Popen(cmd, cwd=str(root_dir), stdout=stream, stderr=subprocess.STDOUT,
                                         creationflags=flags, start_new_session=sys.platform!='win32')
            stream.close()
            worker_log[0] = logpath
            worker_experiment[0] = folder
            displayed_log[0] = None
            train_status.set('Addestramento in corso.')
            header_state.set(f'Training · {folder.name}')
            refresh_experiments(prefer=folder.name)
            exp_name.set(folder.name)
            tabs.select(panels['training'])
        except OSError as exc:
            messagebox.showerror('Cannot start training', str(exc))
    def request_stop():
        if worker[0] and worker[0].poll() is None:
            folder=worker_experiment[0]
            if folder:
                try:
                    (folder/'stop_requested').write_text('stop', encoding='utf-8')
                    train_status.set('Arresto richiesto. Il salvataggio avviene a fine generazione.')
                except OSError as exc:
                    train_status.set('Richiesta di arresto non salvata: '+str(exc))
    ttk.Button(tbar, text='▶ Avvia nuova run', style='Primary.TButton', command=lambda:start_training(False)).pack(side='left', padx=(0,8))
    ttk.Button(tbar, text='↻ Riprendi checkpoint', command=lambda:start_training(True)).pack(side='left', padx=5)
    ttk.Button(tbar, text='■ Ferma e salva', command=request_stop).pack(side='left', padx=5)
    ttk.Button(tbar, text='Aggiorna', command=lambda:refresh_all()).pack(side='left', padx=5)
    ttk.Label(tbar, text='Live campionato; pausa e velocità riguardano solo la visualizzazione.', style='Muted.TLabel').pack(side='right')

    # Arena: actual training frames, replay and independent agent previews.
    toolbar = ttk.Frame(panels['arena'])
    toolbar.pack(fill='x', padx=15, pady=5)
    mode = tk.StringVar(value='live')
    live_buffer = [LiveBuffer()]
    live_exp = [None]
    live_tiles={}
    live_mosaic=tk.BooleanVar(value=False)
    closing = [False]
    replay_brain = [None]
    replay_meta = [{}]
    display_trail = []
    trail_key = [None]
    ttk.Radiobutton(toolbar, text='Training reale', variable=mode, value='live').pack(side='left', padx=4)
    ttk.Radiobutton(toolbar, text='Simulazione agenti', variable=mode, value='preview').pack(side='left', padx=4)
    ttk.Radiobutton(toolbar, text='Replay registrato', variable=mode, value='replay').pack(side='left', padx=4)
    replay_name = tk.StringVar()


    scenario_name=tk.StringVar(value='Preda ferma')

    speed = tk.IntVar(value=130)
    ttk.Label(toolbar, text='Velocità visiva').pack(side='left', padx=(14,4))
    ttk.Scale(toolbar, from_=320, to=45, variable=speed, orient='horizontal', length=120).pack(side='left')
    source_area=ttk.Frame(panels['arena'])
    source_area.pack(fill='x',padx=15,pady=2)
    selection_bar=ttk.Frame(source_area)
    preview_bar=ttk.Frame(source_area)
    replay_bar=ttk.Frame(source_area)
    source_bars={'live':selection_bar,'preview':preview_bar,'replay':replay_bar}
    for bar in source_bars.values():bar.grid(row=0,column=0,sticky='w')
    ttk.Label(preview_bar,text='Scenario').pack(side='left')
    scenario_combo=ttk.Combobox(preview_bar,textvariable=scenario_name,values=list(SCENARIO_LABELS.values()),state='readonly',width=27)
    scenario_combo.pack(side='left',padx=7)
    comparison=tk.StringVar(value='Inseguitore con sensori')
    ttk.Label(preview_bar,text='Controllo di confronto').pack(side='left')
    ttk.Combobox(preview_bar,textvariable=comparison,values=('Inseguitore con sensori','Inseguitore privilegiato'),
                 state='readonly',width=24).pack(side='left',padx=7)
    ttk.Label(replay_bar,text='Episodio registrato').pack(side='left')
    replay_combo=ttk.Combobox(replay_bar,textvariable=replay_name,state='readonly',width=30)
    replay_combo.pack(side='left',padx=7)
    replay_position=tk.IntVar(value=0)
    ttk.Label(replay_bar,text='Fotogramma').pack(side='left')
    def seek_replay(position):
        if mode.get()!='replay' or not replay_frames[0]:return
        index=max(0,min(len(replay_frames[0])-1,round(float(position))))
        if playing[0]:toggle_play()
        replay_index[0]=index
        replay_position.set(index)
        draw_arena();draw_brain()
    timeline=ttk.Scale(replay_bar,from_=0,to=0,variable=replay_position,command=seek_replay,
                       orient='horizontal',length=220)
    timeline.pack(side='left',padx=7)
    ttk.Label(selection_bar,text='Slot (-1 tutti; N = EVA)').pack(side='left')
    live_slot=tk.StringVar(value='-1')
    ttk.Spinbox(selection_bar,from_=-1,to=Limits.MAX_POPULATION,textvariable=live_slot,width=5).pack(side='left',padx=5)
    live_scenario=tk.StringVar(value='all')
    ttk.Combobox(selection_bar,textvariable=live_scenario,values=['all',*ALL_SCENARIOS],state='readonly',width=14).pack(side='left')
    def select_live():
        exp=active_experiment()
        if not exp:return
        try:
            write_mailbox(exp/'live/control.json',{'individual':int(live_slot.get()),'scenario':live_scenario.get(),'stride':1})
            live_buffer[0]=LiveBuffer(individual=int(live_slot.get()),scenario=live_scenario.get())
        except (OSError,ValueError) as exc:arena_status.set('Selezione non disponibile: '+str(exc))
    ttk.Button(selection_bar,text='Applica al training',command=select_live).pack(side='left',padx=5)
    ttk.Button(selection_bar,text='Vai all’ultimo dato',command=lambda:(live_buffer[0].latest(),draw_arena(),draw_brain())).pack(side='left',padx=5)

    canvas = tk.Canvas(panels['arena'], bg='#f8fafc', highlightthickness=1,
                       highlightbackground='#cbd5e1')
    canvas.pack(fill='both', expand=True, padx=15, pady=(6, 2))
    controls = ttk.Frame(panels['arena'])
    controls.pack(fill='x', padx=15, pady=(5, 6))
    playing = [True]
    tick_id = [None]
    preview_actors = []
    scene = [None]
    replay_frames = [[]]
    replay_index = [0]
    arena_step = [0]
    preview_limit = [100]
    legend = ttk.Frame(panels['arena'])
    legend.pack(fill='x', padx=15, pady=(0, 12))
    arena_status = tk.StringVar(value='Choose an experiment; playback starts automatically.')
    ttk.Label(panels['arena'], textvariable=arena_status, style='Muted.TLabel').pack(anchor='w', padx=18, pady=(0, 7))
    visibility = {}
    legend_signature=[None]
    legend_page=[0]
    def legend_step(delta):
        legend_page[0]+=delta
        rebuild_legend()
    def rebuild_legend():
        previous={name:var.get() for name,var in visibility.items()}
        if mode.get()=='live' and live_mosaic.get():
            specs=[(a['name'],a['color']) for a in live_tiles.values()]
        elif mode.get() in ('replay','live'):
            specs = ([(Limits.EVA_LABEL,EVA_COLOR),('Esploratore in training',POPULATION_COLOR)] if mode.get()=='live' else [(Limits.EVA_LABEL,EVA_COLOR)]) + [('Prey','#ef4444')]
        else:
            specs = [(a['name'], a['color']) for a in preview_actors]
        page_size=8
        page_count=max(1,math.ceil(len(specs)/page_size))
        legend_page[0]=max(0,min(page_count-1,legend_page[0]))
        key=(tuple(specs),legend_page[0])
        if key==legend_signature[0]:return
        legend_signature[0]=key
        for w in legend.winfo_children():w.destroy()
        visibility.clear()
        for name,_ in specs:visibility[name]=tk.BooleanVar(value=previous.get(name,True))
        begin=legend_page[0]*page_size
        for index,(name,color) in enumerate(specs[begin:begin+page_size]):
            enabled=visibility[name]
            item=ttk.Frame(legend)
            item.grid(row=index//4,column=index%4,sticky='w',padx=7)
            tk.Label(item,text='●',fg=color,bg=BG).pack(side='left')
            ttk.Checkbutton(item,text=name,variable=enabled,command=lambda:draw_arena()).pack(side='left')
        if page_count>1:
            controls=ttk.Frame(legend)
            controls.grid(row=2,column=0,columnspan=4,sticky='w')
            ttk.Button(controls,text='‹',command=lambda:legend_step(-1)).pack(side='left')
            ttk.Label(controls,text=f'Legenda {legend_page[0]+1}/{page_count}').pack(side='left',padx=8)
            ttk.Button(controls,text='›',command=lambda:legend_step(1)).pack(side='left')
        if not specs:ttk.Label(legend, text='No agents available yet', style='Muted.TLabel').pack(side='left')
    def load_preview():
        preview_actors.clear()
        scene[0]=None
        exp=active_experiment()
        if exp is None:
            arena_status.set('No experiment selected.');rebuild_legend();return
        settings=preview_settings(exp,scenario_id(scenario_name.get()))
        if settings is None:
            arena_status.set('Experiment unavailable or incomplete (config/map missing). Select a valid experiment.')
            rebuild_legend();return
        try:
            scene[0]=settings['map']
            preview_limit[0]=settings['limit']
            rng_seed=settings['seed']*100000+31000
            for i,spec in enumerate(available_agent_specs(exp,'sensor_greedy' if comparison.get()=='Inseguitore con sensori' else 'greedy')):
                w=make_world(settings,rng_seed)
                a=dict(spec,world=w,rng=random.Random(rng_seed+5000+i),steps=0)
                if 'brain' in a:a['brain'].reset()
                preview_actors.append(a)
            arena_step[0]=0
            rebuild_legend()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            preview_actors.clear();scene[0]=None
            arena_status.set('Could not load Arena: '+str(exc));rebuild_legend()
    def load_replay():
        exp=active_experiment()
        replay_frames[0]=[]; replay_index[0]=0;replay_brain[0]=None;replay_meta[0]={}
        timeline.configure(to=0)
        replay_position.set(0)
        if exp is None or not replay_name.get():return
        try:
            item=read_json(exp/'replays'/replay_name.get())
            scene[0]=read_json(exp/'environment'/f'{item["map"]}.json')
            replay_frames[0]=item['frames']
            replay_meta[0]={key:item.get(key) for key in ('seed','scenario','genome_id')}
            timeline.configure(to=max(0,len(replay_frames[0])-1))
            replay_brain[0]=item.get('brain')
            if replay_brain[0] is None and (exp/'brains/eva.json').is_file():
                candidate=read_json(exp/'brains/eva.json')
                from engine.evaluation import genome_id
                if genome_id(BooleanNet.restore(candidate))==item.get('genome_id'):
                    replay_brain[0]=candidate
            rebuild_legend()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            arena_status.set('Cannot load replay: '+str(exc))
    def update_mode(*_):
        for name,bar in source_bars.items():
            if name==mode.get():bar.grid()
            else:bar.grid_remove()
        if mode.get()=='live':
            scene[0]=None;rebuild_legend()
        elif mode.get()=='replay':load_replay()
        else:load_preview()
        playing[0]=True
        play_button.configure(text='Pausa')
        if tick_id[0] is None:tick_id[0]=app.after(max(35,int(speed.get())),schedule_tick)
        draw_arena();draw_brain()
    def arena_positions():
        if mode.get()=='live':
            current=live_buffer[0].current
            if not current:return None,[],live_buffer[0].label()
            packet,f=current
            actor=Limits.EVA_LABEL if packet.get('phase')=='eva' else f'Esploratore {packet.get("individual",0)+1}'
            label=(f'Live Training · {actor} · gen {packet.get("generation","—")} · {friendly(packet.get("scenario","—"))}'
                   f' · genoma {str(packet.get("genome_id","—"))[:10]} · seed {packet.get("seed","—")}\n'
                   f'{frame_summary(f)} · {live_buffer[0].label()}')
            return packet['map'],[{'name':Limits.EVA_LABEL if packet.get('phase')=='eva' else 'Esploratore in training','color':EVA_COLOR if packet.get('phase')=='eva' else POPULATION_COLOR,'pos':tuple(f['predator'])},
                                  {'name':'Prey','color':'#ef4444','pos':tuple(f['prey'])}],label
        if mode.get()=='replay':
            if not replay_frames[0]:return None,[],''
            f=replay_frames[0][replay_index[0]]
            actors=[{'name':Limits.EVA_LABEL,'color':EVA_COLOR,'pos':tuple(f['predator'])}]
            actors.append({'name':'Prey','color':'#ef4444','pos':tuple(f['prey'])})
            meta=replay_meta[0]
            label=(f'Recorded Replay · EVA · {friendly(meta.get("scenario","—"))} · seed {meta.get("seed","—")}'
                   f' · genoma {str(meta.get("genome_id","—"))[:10]} · frame {replay_index[0]+1}/{len(replay_frames[0])}\n'
                   +frame_summary(f))
            return scene[0],actors,label
        if not preview_actors:return None,[],''
        actors=[{'name':a['name'],'color':a['color'],'pos':a['world'].predator} for a in preview_actors]
        actors.append({'name':'Prey','color':'#ef4444','pos':preview_actors[0]['world'].prey})
        captures=sum(a['world'].captured for a in preview_actors)
        label=f'Independent Preview · step {arena_step[0]} · captured {captures}/{len(preview_actors)} · independent same-seed arenas (not live training frames)'
        return scene[0],actors,label
    def draw_mosaic(source=None):
        actors=[a for a in (preview_actors if source is None else source) if visibility.get(a['name']) is None or visibility[a['name']].get()]
        if not actors:
            arena_status.set('Nessun agente visibile. Seleziona un agente nella legenda.')
            return
        width=max(300,canvas.winfo_width());height=max(300,canvas.winfo_height())
        cols=max(1,math.ceil(math.sqrt(len(actors)*width/height)))
        rows=math.ceil(len(actors)/cols);tw=width/cols;th=height/rows
        for i,a in enumerate(actors):
            world=a['world'];m=a.get('map',scene[0]);size=m['size']
            cell=max(1,min((tw-16)/size,(th-30)/size));ox=(i%cols)*tw+8;oy=(i//cols)*th+26
            canvas.create_text(ox,oy-14,text=a['name']+(f' · g{a["generation"]} · {friendly(a["scenario"])}' if 'generation' in a else ''),anchor='w',fill=a['color'],font=('Segoe UI',8))
            obstacles={tuple(p) for p in m['obstacles']}
            for y in range(size):
                for x in range(size):
                    canvas.create_rectangle(ox+x*cell,oy+y*cell,ox+(x+1)*cell,oy+(y+1)*cell,
                                            fill='#334155' if (x,y) in obstacles else 'white',outline='#e2e8f0')
            for position,color in [(world.prey,'#ef4444'),(world.predator,a['color'])]:
                x,y=position;r=cell*.34;cx=ox+(x+.5)*cell;cy=oy+(y+.5)*cell
                canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill=color,outline='white')
        if source is not None:
            arena_status.set('Ultime osservazioni reali per slot (massimo 32), raccolte in momenti diversi; non sono agenti simultanei.')
            return
        arena_status.set(f'Simulazioni indipendenti · {scenario_name.get()} · passo {arena_step[0]} · '
                         f'catture {sum(a["world"].captured for a in actors)}/{len(actors)} agenti visibili · preda rossa per ciascun mondo')

    def draw_arena():
        canvas.delete('all')
        if mode.get()=='live' and live_mosaic.get() and live_tiles:
            draw_mosaic(list(live_tiles.values()));return
        if mode.get()=='preview' and preview_actors:
            draw_mosaic();return
        m, agents, label=arena_positions()
        if not m:
            arena_status.set(label or 'Select an experiment with available data.')
            return
        size=m['size'];w=max(300, canvas.winfo_width());h=max(300, canvas.winfo_height())
        cell=min((w-35)/size,(h-35)/size)
        ox=(w-size*cell)/2;oy=(h-size*cell)/2
        obstacles={tuple(p) for p in m['obstacles']}
        for y in range(size):
            for x in range(size):
                canvas.create_rectangle(ox+x*cell,oy+y*cell,ox+(x+1)*cell,oy+(y+1)*cell,
                                        fill='#334155' if (x,y) in obstacles else '#ffffff',outline='#cbd5e1')
        if mode.get()=='live' and live_buffer[0].current:
            packet,frame=live_buffer[0].current
            key=(packet.get('run_id'),packet.get('episode_id'))
            if trail_key[0]!=key:display_trail.clear();trail_key[0]=key
            # Draw only consecutive, actually captured positions; never interpolate a gap.
            trace=[point for point in packet.get('trajectory',[]) if point['t']<=frame['t']]
            for before,after in zip(trace,trace[1:]):
                if after['t']==before['t']+1:
                    x0,y0=before['position'];x1,y1=after['position']
                    canvas.create_line(ox+(x0+.5)*cell,oy+(y0+.5)*cell,
                                       ox+(x1+.5)*cell,oy+(y1+.5)*cell,fill='#93c5fd',width=2)
        positions={}
        for a in agents:
            if a['name'] != 'Prey' and a['name'] in visibility and not visibility[a['name']].get():continue
            if a['name']=='Prey' and mode.get() in ('replay','live') and 'Prey' in visibility and not visibility['Prey'].get():continue
            positions.setdefault(a['pos'], []).append(a)
        for (x,y),group in positions.items():
            for idx,a in enumerate(group):
                theta=2*math.pi*idx/len(group)
                r=cell*.24 if len(group)>1 else 0
                cx=ox+(x+.5)*cell+r*math.cos(theta);cy=oy+(y+.5)*cell+r*math.sin(theta)
                radius=min(cell*.22,cell*.4/max(1,math.sqrt(len(group))))
                canvas.create_oval(cx-radius,cy-radius,cx+radius,cy+radius,fill=a['color'],outline='white',width=2)
                canvas.create_text(cx,cy,text='P' if a['name']=='Prey' else 'E',fill='white',
                                   font=('Segoe UI',max(7,int(radius*.8)),'bold'))
        arena_status.set(label)
    def next_step():
        if mode.get()=='live':
            if playing[0]:live_buffer[0].latest()
            else:live_buffer[0].advance()
        elif mode.get()=='replay':
            if replay_frames[0]:
                if replay_index[0] + 1 >= len(replay_frames[0]):return False
                replay_index[0]+=1
                replay_position.set(replay_index[0])
        elif preview_actors:
            if episode_finished(preview_actors,preview_limit[0]):
                load_preview()   # Automatically restart; safe if source was removed.
            else:
                limit=preview_limit[0]
                for a in preview_actors:
                    if a['world'].captured or a['world'].steps>=limit:continue
                    obs=a['world'].observe()
                    previous=list(a['brain'].state) if 'brain' in a else []
                    action=preview_action(a)
                    reward,captured=a['world'].step(action)
                    if 'brain' in a:
                        brain=a['brain']
                        if hasattr(brain,'transition'):brain.transition(a['world'].observe(),captured)
                        if hasattr(brain,'feedback'):brain.feedback(reward)
                        a['frame']={'t':a['world'].steps,'action':action,'observation':list(obs),
                                    'previous_state':previous,'neural_state':list(brain.state)}
                        if hasattr(a['world'],'world_events'):a['frame']['world_events']=a['world'].world_events()
                        if hasattr(brain,'frame_data'):a['frame'].update(brain.frame_data())
                        if hasattr(brain,'tables'):
                            a['frame'].update(plastic_tables=[list(t) for t in brain.tables],
                                              plastic_updates=list(brain.last_updates),plastic_update_count=brain.update_count)
                arena_step[0]+=1
        else:return False
        draw_arena()
        draw_brain()
        return True
    def schedule_tick():
        tick_id[0]=None
        if closing[0] or not playing[0]:return
        if mode.get()=='replay' and replay_frames[0] and replay_index[0]==len(replay_frames[0])-1:
            restart()
        elif mode.get()!='live':
            next_step()
        tick_id[0]=app.after(max(35,int(speed.get())),schedule_tick)
    def toggle_play():
        playing[0]=not playing[0]
        play_button.configure(text='Pausa' if playing[0] else 'Riprendi')
        if playing[0] and tick_id[0] is None:schedule_tick()
    def step_observed():
        if playing[0]:toggle_play()
        next_step()
    def restart():
        if mode.get()=='live':live_buffer[0].latest()
        elif mode.get()=='replay':replay_index[0]=0;replay_position.set(0)
        else:load_preview()
        draw_arena();draw_brain()
    play_button=ttk.Button(controls, text='Pausa', command=toggle_play)
    play_button.pack(side='left',padx=4)
    ttk.Button(controls,text='Un passo',command=lambda:step_observed()).pack(side='left',padx=4)
    ttk.Button(controls,text='↻ Ricomincia / Ultimo dato',command=restart).pack(side='left',padx=4)
    ttk.Button(controls,text='Mostra tutti',command=lambda:([v.set(True) for v in visibility.values()],draw_arena())).pack(side='left',padx=4)
    ttk.Button(controls,text='Nascondi tutti',command=lambda:([v.set(False) for v in visibility.values()],draw_arena())).pack(side='left',padx=4)
    ttk.Checkbutton(selection_bar,text='Mosaico live',variable=live_mosaic,command=lambda:(rebuild_legend(),draw_arena())).pack(side='left',padx=4)
    canvas.bind('<Configure>',lambda _:draw_arena())
    for name,bar in source_bars.items():
        if name!=mode.get():bar.grid_remove()
    mode.trace_add('write',update_mode)
    replay_name.trace_add('write',lambda *_: (load_replay(),draw_arena(),draw_brain()) if mode.get()=='replay' else None)
    comparison.trace_add('write',lambda *_: (load_preview(),draw_arena(),draw_brain()) if mode.get()=='preview' else None)
    scenario_name.trace_add('write',lambda *_: (load_preview(),draw_arena()) if mode.get()=='preview' else None)

    ttk.Button(controls,text='Guida ai controlli',command=lambda:messagebox.showinfo('Arena',
        'Training reale: osserva esploratori ed EVA mentre apprendono. Con N esploratori, lo slot N è EVA. Slot -1 accetta tutti; Applica imposta il filtro.\n'
        'Simulazione agenti: confronta campioni, popolazione e controlli in arene indipendenti con lo stesso seed. Le caselle mostrano/nascondono gli agenti.\n'
        'Replay: riproduce un episodio registrato. Pausa, passo e velocità cambiano solo la visualizzazione.\n'
        'static = preda ferma; moving = preda mobile; maze = ostacoli; hidden = visibilità ridotta a 3 celle, NON preda invisibile.\n'
        'Random usa 4 movimenti; Greedy conosce la posizione vera della preda. I confronti non hanno tutti gli stessi sensori.')).pack(side='left')

    # Evolution and benchmark measurements use the same agent labels and palette.
    graph_scenario=tk.StringVar(value='Tutti gli scenari')
    ttk.Combobox(panels['evolution'],textvariable=graph_scenario,values=['Tutti gli scenari',*SCENARIO_LABELS.values()],state='readonly').pack(anchor='w',padx=15,pady=3)
    graph_metric=tk.StringVar(value='Catture')
    ttk.Combobox(panels['evolution'],textvariable=graph_metric,values=['Catture','Passi per cattura','Neuroni attivi'],state='readonly').pack(anchor='w',padx=15,pady=5)
    plot = tk.Canvas(panels['evolution'], bg=PANEL, highlightthickness=1, highlightbackground='#cbd5e1')
    plot.pack(fill='both',expand=True,padx=15,pady=15)
    graph_caption = ttk.Label(panels['evolution'], text='Blu: EVA - learning agent · Verde: Popolazione — media', wraplength=1100, style='Muted.TLabel')
    graph_caption.pack(anchor='w',padx=17,pady=(0,12))
    benchmark_note=tk.StringVar()
    ttk.Label(panels['benchmark'],textvariable=benchmark_note,style='Muted.TLabel').pack(anchor='w',padx=15,pady=6)
    benchmark_area=ttk.Frame(panels['benchmark'])
    benchmark_area.pack(fill='both',expand=True,padx=15,pady=(0,12))
    table = ttk.Treeview(benchmark_area, columns=('split','capture','count','wilson','steps','median','stall','neurons','blocked'),show='headings')
    for col, label, wd in [('split','Agente / split / scenario',330),('capture','Catture',76),('count','Successi/n',78),
                           ('wilson','IC 95%',112),('steps','Passi medi¹',87),('median','Mediana¹',79),('stall','Stalli',75),
                           ('neurons','Moduli',65),('blocked','Blocchi medi',94)]:
        table.heading(col,text=label)
        table.column(col,anchor='w' if col=='split' else 'center',width=wd,minwidth=65)
    scrollbar=ttk.Scrollbar(benchmark_area,orient='vertical',command=table.yview)
    scrollbar.pack(side='right',fill='y')
    scroll_x=ttk.Scrollbar(benchmark_area,orient='horizontal',command=table.xview)
    scroll_x.pack(side='bottom',fill='x')
    table.configure(yscrollcommand=scrollbar.set,xscrollcommand=scroll_x.set)
    table.pack(fill='both',expand=True)
    ttk.Label(panels['benchmark'],text='¹ Solo catture riuscite. IC di Wilson per gli episodi del singolo scenario. Trattino = misura assente.',
              style='Muted.TLabel').pack(anchor='w',padx=18,pady=(0,10))

    # Inspector consumes precisely the Arena's selected source/frame.
    brain_controls=ttk.Frame(panels['brain']);brain_controls.pack(fill='x',padx=12,pady=4)
    ttk.Button(brain_controls,text='Pausa / Riprendi',command=toggle_play).pack(side='left')
    ttk.Button(brain_controls,text='Passo osservato',command=step_observed).pack(side='left')
    ttk.Label(brain_controls,text='Sorgente e selezione condivise con Arena').pack(side='left',padx=10)
    inspector=BrainInspector(panels['brain']);inspector.pack(fill='both',expand=True)
    brain_choice=tk.StringVar()
    brain_combo=ttk.Combobox(brain_controls,textvariable=brain_choice,state='readonly',width=42)
    brain_combo.pack(side='left',padx=6)
    brain_combo.bind('<<ComboboxSelected>>',lambda _:draw_brain())
    def draw_brain():
        brain_combo.configure(state='readonly' if mode.get()=='preview' else 'disabled')
        if mode.get()=='live':
            current=live_buffer[0].current
            if current:
                packet,frame=current
                brain_choice.set(Limits.EVA_LABEL if packet.get('phase')=='eva' else f'Esploratore {packet.get("individual",0)+1}')
                inspector.show(packet.get('brain'),frame,
                               f'Live Training / gen {packet.get("generation")} / slot {packet.get("individual")} / {packet.get("scenario")}')
            else:inspector.show(None,source='Live non disponibile')
        elif mode.get()=='replay' and replay_frames[0]:
            brain_choice.set(Limits.EVA_LABEL)
            inspector.show(replay_brain[0],replay_frames[0][replay_index[0]],'Recorded Replay')
        elif mode.get()=='preview' and preview_actors:
            brains=[a for a in preview_actors if 'brain' in a]
            brain_combo['values']=[a['name'] for a in brains]
            actor=next((a for a in brains if a['name']==brain_choice.get()),brains[0] if brains else None)
            if actor:
                brain_choice.set(actor['name'])
                inspector.show(actor['brain'].export(),actor.get('frame'),actor['name']+' / Independent Preview')
            else:inspector.show(None,source='Independent Preview')
        else:inspector.show(None,source='Nessun dato sincronizzato')
    diagnostics_text=tk.Text(panels['diagnostics'],wrap='word',font=('Consolas',10))
    diagnostics_text.pack(fill='both',expand=True,padx=15,pady=15)
    def refresh_diagnostics():
        exp=active_experiment()
        text=f'Esperimento: {exp.name if exp else "nessuno"} · Predatore {Limits.VERSION}\n'
        text+='La diagnostica viene prodotta al termine del training.\n'
        text+='Confronti appaiati: normale, memoria OFF; nel vettoriale anche RAM OFF e conoscenze cancellate.\n'
        text+='Per ablazioni di ciascun neurone: python launcher.py --diagnose PERCORSO_ESPERIMENTO\n\n'
        if exp:
            try:
                report=read_json(exp/'neuroscience/behavioral_diagnostics.json')
                text+=report['protocol']+'\n'+report['note']+'\n\n'
                text+=f"Nodi con percorso strutturale alle uscite: {report['output_ancestors']}\n\n"
                for name,item in report['variants'].items():
                    text+=(f"{name}: catture {item['capture_rate']:.1%}; Δ normale {item['capture_delta_vs_normal']:+.1%}; "
                           f"celle uniche {item['mean_unique_cells']:.1f}; modifiche plastiche {item['plastic_updates']}\n")
                    for scenario in dict.fromkeys(r['scenario'] for r in item['episodes']):
                        rows=[r for r in item['episodes'] if r['scenario']==scenario]
                        if rows:text+=f"  {scenario}: catture {sum(r['capture'] for r in rows)}/{len(rows)}; conteggi azioni { [sum(r['action_counts'][i] for r in rows) for i in range(8)] }\n"
            except (OSError,ValueError,KeyError,TypeError):text+='Nessun report disponibile per questo esperimento.'
        if exp:
            try:
                profile,state=selected_state()
                if state:
                    text+='\n'+Limits.SMITH_LABEL+' / '+str(profile)+'\n'+json.dumps(state['meta'],indent=2,ensure_ascii=False)+'\n'
            except (OSError,ValueError,KeyError):pass
        if exp:
            reports=list((root_dir/'Analysis').glob(f'{exp.name}_probes_*/supplementary.json'))
            if reports:
                latest=max(reports,key=lambda p:p.stat().st_mtime_ns)
                try:
                    supplementary=read_json(latest)
                    text+='\nPROVE SUPPLEMENTARI — nessun training\n'+supplementary['protocol']['note']+'\n'
                    for name,results in supplementary['scenarios'].items():
                        text+='\n'+name+'\n'
                        for actor,row in results.items():
                            text+=f'  {actor}: {row["capture_rate"]:.1%}; salti legali {row["pounce_legal_moves"]}/{row["pounce_attempts"]} tentativi\n'
                except (OSError,ValueError,KeyError,TypeError):text+='\nReport supplementare non leggibile.'
        diagnostics_text.delete('1.0','end');diagnostics_text.insert('end',text)

    ttk.Label(panels['supervisor'],text=Limits.SMITH_LABEL,font=('Segoe UI',13,'bold')).pack(anchor='w',padx=18,pady=12)
    ttk.Label(panels['supervisor'],text='Conserva esperienze, specialisti e risultati delle strategie tra le run. EVA è il suo agente che apprende e agisce.',wraplength=1050).pack(anchor='w',padx=18)
    supervisor_text=tk.Text(panels['supervisor'],wrap='word',font=('Consolas',10))
    supervisor_text.pack(fill='both',expand=True,padx=18,pady=12)
    def refresh_supervisor():
        text=Limits.SMITH_LABEL+' — nessun dato per la run selezionata.'
        try:
            profile,state=selected_state()
            if state is None:raise ValueError('No state')
            meta=state['meta']
            text=Limits.SMITH_LABEL+'\nProfilo SMITH: '+profile+' · revisione '+str(state['generation'])+'\n'
            text+='Run visualizzata: '+state['transaction']+'\n'
            text+=f"Archivio: {len(state['events'])} esperienze · {len(state['explorers'])} specialisti · {len(state['tasks'])} compiti di riferimento\n"
            text+='Strategia: '+str(meta.get('last_strategy','—'))+' · contesto binario '+format(meta.get('last_context',0),'06b')+'\n'
            text+='Bit: poche catture, molti stalli, trasferimento respinto, conflitto di memoria, moduli sottoutilizzati, regressione dopo apprendimento.\n'
            text+='Statistiche: tentativi / miglioramenti del punteggio bilanciato in validazione.\n\n'
            history=state.get('history') or []
            if history and 'learning_channels' in history[-1]:
                last=history[-1]
                text+=f'Apprendimento personale EVA: {"ripristino" if last.get("personal_rollback") else "conservato"}'
                text+=f' · candidato {last.get("personal_candidate_id","—")}\n'
                for channel,counts in last['learning_channels'].items():
                    text+=f'{channel}: {counts["accepted"]} accettate, {counts["rejected"]} rifiutate\n'
                text+='\n'
            text+=json.dumps(meta,indent=2,ensure_ascii=False)
        except (OSError,ValueError,KeyError,TypeError):pass
        supervisor_text.delete('1.0','end');supervisor_text.insert('end',text)

    def format_rate(value):return '—' if value is None else f'{value:.1%}'

    def add_measurement(label,row,neurons=None):
        low_high=row.get('capture_wilson95')
        interval=f'{low_high[0]:.0%}–{low_high[1]:.0%}' if low_high else '—'
        count=(f'{row["capture_count"]}/{row["episode_count"]}'
               if 'capture_count' in row and 'episode_count' in row else '—')
        def number(key):
            value=row.get(key)
            return f'{value:.1f}' if value is not None else '—'
        table.insert('','end',values=(label,format_rate(row.get('capture_rate')),count,interval,
            number('successful_mean_steps'),number('successful_median_steps'),format_rate(row.get('stall_rate')),
            row.get('active_neurons',neurons if neurons is not None else '—'),number('blocked_actions')))

    last_refresh = [None]
    selected_source = [None]
    def refresh_experiments(prefer=None):
        # Ignore orphan folders containing only logs; they cannot be replayed.
        folders=[p for p in vault.glob('EXP_*') if p.is_dir() and (p/'config.json').is_file()]
        folders.sort(key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)
        names=[p.name for p in folders]
        # A newly launched worker may not have written config.json yet. Keep its selection.
        pending=worker_experiment[0]
        if worker[0] is not None and pending is not None and pending.name not in names:
            names.insert(0,pending.name)
        exp_combo['values']=names
        chosen=prefer if prefer in names else (exp_name.get() if exp_name.get() in names else (names[0] if names else ''))
        if exp_name.get()!=chosen:
            exp_name.set(chosen)
            if not chosen:
                preview_actors.clear();scene[0]=None
                live_buffer[0]=LiveBuffer();live_exp[0]=None
                arena_status.set('No valid experiment found; config.json is missing.')
                rebuild_legend()
        return chosen
    def load_selected(*_):
        exp=active_experiment()
        if selected_source[0]!=exp:
            selected_source[0]=exp
            last_refresh[0]=None;loaded_checkpoint[0]=None;displayed_log[0]=None
            preview_actors.clear();scene[0]=None;replay_frames[0]=[];replay_brain[0]=None
            replay_index[0]=0;arena_step[0]=0;logs.delete('1.0','end')
            plot.delete('all')
            benchmark_note.set('')
            for item in table.get_children():table.delete(item)
            train_status.set('In attesa dei dati della run '+(exp.name if exp else '—'))
            inspector.show(None,source='Nuova selezione')
        if live_exp[0]!=exp:
            live_buffer[0]=LiveBuffer();live_exp[0]=exp;live_tiles.clear()
            display_trail.clear();trail_key[0]=None
        if not exp:
            refresh_diagnostics();draw_arena();draw_brain();return
        replays=sorted(p.name for p in (exp/'replays').glob('*.json'))
        replay_combo['values']=replays
        if replay_name.get() not in replays:replay_name.set(replays[0] if replays else '')
        available_scenarios=[name for name in ALL_SCENARIOS
                            if (exp/'environment'/'suites'/f'holdout_{name}.json').is_file()]
        scenario_combo['values']=[friendly(n) for n in available_scenarios] or ['Preda ferma']
        if available_scenarios and scenario_id(scenario_name.get()) not in available_scenarios:
            scenario_name.set(friendly(available_scenarios[0]))
        update_mode()
        last_refresh[0]=None
        refresh_data(force=True)
        refresh_eva();refresh_supervisor()
        refresh_diagnostics()
    def refresh_data(force=False):
        exp=active_experiment()
        if not exp:return
        if not (exp/'config.json').is_file():return
        cp=latest_checkpoint_path(exp/'checkpoints');benchmark=exp/'benchmarks/results.json';brain=exp/'neuroscience/brain_graph.json'
        try:
            watched=[cp,benchmark,brain,exp/'monitor/latest.json',exp/'monitor/history.json',
                     exp/'benchmarks/population_comparison.json',exp/'benchmarks/baselines.json',
                     exp/'neuroscience/behavioral_diagnostics.json',exp/'config.json']
            stamp=(str(exp.resolve()),tuple((str(path),path.stat().st_mtime_ns,path.stat().st_size)
                                           for path in watched if path is not None and path.exists()))
            if not force and stamp==last_refresh[0]:return
            if cp is not None and cp.exists():
                history=read_json(cp).get('history',[])
                if history:
                    newest=history[-1]
                    train_status.set(f'Generazione {newest["generation"]} | {Limits.EVA_LABEL}: {newest["capture_rate"]:.1%} | Popolazione: {newest["population_mean_capture"]:.1%} | Moduli EVA: {newest.get("executed_neurons","—")}/{int(newest["active_neurons"])} | Decodifica: {newest.get("decoder_counts",{})}')
                else:history=[]
            else:history=[]
            monitor_path=exp/'monitor/latest.json'
            monitor=read_json(monitor_path) if monitor_path.is_file() else {}
            if monitor.get('train') and (not history or monitor['generation']>history[-1]['generation']):
                history=history+[monitor['train']]
            from gui.progress_chart import draw_progress
            validation_path=exp/'monitor/history.json'
            validation=read_json(validation_path) if validation_path.is_file() else []
            selected=scenario_id(graph_scenario.get())
            chart_baselines=monitor.get('baselines',{})
            if selected!='Tutti gli scenari':
                history=[{'generation':r['generation'],'active_neurons':r.get('active_neurons'),'population_mean_neurons':r.get('population_mean_neurons'),
                          **r.get('by_scenario',{}).get(selected,{}),
                          **r.get('comparison_by_scenario',{}).get(selected,{})} for r in history]
                validation=[{'generation':r['generation'],'validation':r['validation'].get('by_scenario',{}).get(selected,{})} for r in validation]
                chart_baselines={split:{name:row.get('by_scenario',{}).get(selected,{}) for name,row in items.items()} for split,items in chart_baselines.items()}
            graph_caption.configure(text=draw_progress(plot,history,graph_metric.get(),validation,chart_baselines))
            for id in table.get_children():table.delete(id)
            cfg=read_json(exp/'config.json')
            benchmark_note.set(f'Seed {cfg.get("seed","—")} · mappa {cfg.get("size","—")} × {cfg.get("size","—")} · '
                               f'budget {cfg.get("steps","—")} passi/episodio · curriculum {cfg.get("scenario_set","classic")}')
            for split in ('train','validation'):
                if split in monitor:
                    row=monitor[split]
                    generation=monitor.get('validation_generation' if split=='validation' else 'generation','—')
                    label=f'EVA / {friendly(split)} · gen {generation}'
                    add_measurement(label,row)
                    for name,detail in row.get('by_scenario',{}).items():
                        add_measurement(f'{label} / {friendly(name)}',detail)
                for name,row in monitor.get('baselines',{}).get(split,{}).items():
                    add_measurement(f'{friendly(name)} / {friendly(split)}',row)
            if benchmark.exists():
                for split,summary in read_json(benchmark).items():
                    label=f'EVA / {friendly(split)}'
                    add_measurement(label,summary)
                    for name,detail in summary.get('by_scenario',{}).items():
                        add_measurement(f'{label} / {friendly(name)}',detail,summary.get('active_neurons'))
            baselines=exp/'benchmarks'/'baselines.json'
            if baselines.is_file():
                for name,row in read_json(baselines).items():
                    label=f'{friendly(name)} / test finale'
                    add_measurement(label,row)
                    for scenario,detail in row.get('by_scenario',{}).items():
                        add_measurement(f'{label} / {friendly(scenario)}',detail)
            comparison_path=exp/'benchmarks/population_comparison.json'
            if comparison_path.is_file():
                comparison=read_json(comparison_path)
                for label,key in [('Popolazione · media','population_mean_capture'),
                                  ('Popolazione · migliore','population_best_capture')]:
                    add_measurement(label+' / test finale',{'capture_rate':comparison[key]})
            if brain.exists():draw_brain()
            last_refresh[0]=stamp
            refresh_diagnostics()
        except (OSError,ValueError,KeyError,TypeError) as exc:
            train_status.set('Could not refresh experiment: '+str(exc))
    graph_scenario.trace_add('write',lambda *_:refresh_data(force=True))
    graph_metric.trace_add('write',lambda *_:refresh_data(force=True))
    plot.bind('<Configure>',lambda _:refresh_data(force=True))
    def refresh_all():
        refresh_eva();refresh_supervisor()
        previous=exp_name.get()
        refresh_experiments()
        if exp_name.get()!=previous:load_selected()
        refresh_data()
    def periodic():
        if closing[0]:return
        refresh_eva();refresh_supervisor()
        try:
            previous=exp_name.get()
            refresh_experiments()
            if exp_name.get()!=previous:load_selected()
            refresh_data()
            active = worker[0] and worker[0].poll() is None
            if active:
                exp=active_experiment()
                cp=latest_checkpoint_path(exp/'checkpoints') if exp else None
                if cp and cp.exists():
                    m=cp.stat().st_mtime_ns
                    if loaded_checkpoint[0]!=(str(exp),m) and mode.get()=='preview':
                        loaded_checkpoint[0]=(str(exp),m)
                        load_preview()
                if worker_log[0] and worker_log[0].exists():
                    blob=worker_log[0].read_bytes()[-8000:].decode('utf-8',errors='replace')
                    if blob!=displayed_log[0]:
                        displayed_log[0]=blob
                        logs.delete('1.0','end');logs.insert('end',blob);logs.see('end')
                refresh_data()
            elif worker[0] is not None:
                code=worker[0].returncode
                header_state.set(f'Addestramento terminato · codice {code}')
                worker[0]=None
                load_selected()
                refresh_data(force=True)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            train_status.set('Aggiornamento temporaneamente non disponibile: '+str(exc))
        if not closing[0]:app.after(1500,periodic)
    def on_tab_change(*_):
        if tabs.select()==str(panels['arena']):draw_arena()
        if tabs.select()==str(panels['results']):refresh_data(force=True)
        if tabs.select()==str(panels['brain']):draw_brain()
        if tabs.select()==str(panels['archive']) and notebooks['archive'].select()==str(panels['diagnostics']):refresh_diagnostics()
        if tabs.select()==str(panels['archive']) and notebooks['archive'].select()==str(panels['supervisor']):refresh_supervisor()
    tabs.bind('<<NotebookTabChanged>>',on_tab_change)
    for notebook in notebooks.values():notebook.bind('<<NotebookTabChanged>>',on_tab_change)
    exp_name.trace_add('write',load_selected)
    exp_combo.bind('<<ComboboxSelected>>',load_selected)
    # Load all previous experiment info; begin playback without asking the user to press Play.
    refresh_experiments()
    if exp_name.get():load_selected()
    draw_arena()
    schedule_tick()
    periodic()
    def poll_live():
        if closing[0]:return
        exp=active_experiment()
        if exp and mode.get()=='live':
            changed=live_buffer[0].poll(exp)
            if not live_buffer[0].current and playing[0]:live_buffer[0].advance()
            if playing[0]:
                live_buffer[0].latest()
                current=live_buffer[0].current
                if current:
                    packet,frame=current;slot=packet.get('individual',0)
                    if slot not in live_tiles and len(live_tiles)>=32:live_tiles.pop(next(iter(live_tiles)))
                    live_tiles[slot]={'name':Limits.EVA_LABEL if packet.get('phase')=='eva' else f'Esploratore {slot+1}',
                        'color':EVA_COLOR if packet.get('phase')=='eva' else agent_color(slot),
                        'world':SimpleNamespace(predator=frame['predator'],prey=frame['prey']),
                        'map':packet['map'],'generation':packet.get('generation'),'scenario':packet.get('scenario')}
                    if live_mosaic.get():rebuild_legend()
                if changed or not current:
                    draw_arena();draw_brain()
                elif not live_mosaic.get():
                    arena_status.set(arena_positions()[2])
        app.after(100,poll_live)
    def close_window():
        closing[0]=True
        # Detached training continues; the mailbox writer never depends on this window.
        for callback in app.tk.call('after','info'):
            try:app.after_cancel(callback)
            except tk.TclError:pass
        app.destroy()
    app.protocol('WM_DELETE_WINDOW',close_window)
    poll_live()
    if test_hook:test_hook(app,locals())
    app.mainloop()

if __name__ == '__main__':
    launch()
