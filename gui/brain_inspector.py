from engine.settings import Limits
"""Reusable Tk observer. Nodes/edges persist while only state colors change."""
import json
import math
import tkinter as tk
from tkinter import ttk, filedialog
from engine.boolean_net import GATES, gate

SENSORS = ('Preda visibile', 'Preda N', 'Preda S', 'Preda O', 'Preda E',
           'Libero N1', 'Libero S1', 'Libero O1', 'Libero E1',
           'Libero N2', 'Libero S2', 'Libero O2', 'Libero E2', 'Preda vicina', 'Collisione avvenuta')


class BrainInspector(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.info = tk.StringVar(value='Nessun cervello osservato')
        ttk.Label(self, textvariable=self.info, wraplength=1100).pack(anchor='w', padx=10, pady=5)
        ttk.Button(self, text='Confronta con cervello JSON…', command=self.compare).pack(anchor='w', padx=10)
        area = ttk.Frame(self)
        area.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(area, bg='#ffffff', highlightthickness=0)
        sy = ttk.Scrollbar(area, orient='vertical', command=self.canvas.yview)
        sx = ttk.Scrollbar(area, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side='right', fill='y');sx.pack(side='bottom', fill='x')
        self.canvas.pack(fill='both', expand=True)
        self.details = tk.Text(self, height=6, wrap='word', font=('Consolas', 10))
        self.details.pack(fill='x', padx=8, pady=5)
        self.signature = None
        self.brain = None
        self.frame = {}
        self.node_items = []
        self.sensor_items = []
        self.gate_labels = []
        self.selected = None
        self.rebuild_count = 0

    def show(self, brain, frame=None, source='Struttura salvata'):
        if not brain:
            self.info.set(source + ' · stati neurali non disponibili')
            self.canvas.delete('all');self.signature=None;self.brain=None
            return
        frame = frame or {}
        self.brain, self.frame = brain, frame
        signature = json.dumps({k:brain.get(k) for k in ('sensors','nodes','outputs','learning','shape','rules','output_lanes')},sort_keys=True)
        if signature != self.signature:
            self.signature = signature
            self._build()
        states = frame.get('neural_state', [])
        observation = frame.get('observation', [])
        tables = [list(row) for row in frame.get('plastic_tables', [])]
        for update in frame.get('plastic_updates', []):
            if update['neuron'] < len(tables):tables[update['neuron']][update['entry']] = update['before']
        for i, item in enumerate(self.node_items):
            n = brain['nodes'][i]
            color = '#94a3b8' if not n['active'] else ('#22c55e' if i<len(states) and states[i] else '#1e3a5f')
            self.canvas.itemconfigure(item, fill=color)
            if i<len(tables):
                effective=next((op for op in GATES if [gate(op,a,b) for a,b in ((0,0),(0,1),(1,0),(1,1))]==tables[i]),'TABLE')
                self.canvas.itemconfigure(self.gate_labels[i],text=effective+(' *' if effective!=n['op'] else ''))
        for i, item in enumerate(self.sensor_items):
            self.canvas.itemconfigure(item, fill='#fbbf24' if i<len(observation) and observation[i] else '#e2e8f0')
        state_note = 'ON verde / OFF blu / disabilitato grigio' if states else 'Solo struttura: attivazioni assenti'
        self.info.set(f'{source} · passo {frame.get("t","—")} · azione {frame.get("action","—")} · '
                      f'{sum(n["active"] for n in brain["nodes"])}/{len(brain["nodes"])} abilitati · '
                      f'aggiornamenti plastici {frame.get("plastic_update_count",0)} · {state_note}')
        if brain.get('engine')==Limits.ENGINE:
            width=frame.get('vector_width',math.prod(brain['shape']))
            for label in self.gate_labels:self.canvas.itemconfigure(label,text=f'LUT {width}b')
            self.info.set(f'{source} · passo {frame.get("t","—")} · {len(brain["nodes"])} moduli × {width} bit · azione RAM {frame.get("action","—")} · memorie {frame.get("knowledge_entries",len(brain.get("knowledge",[])))} · colori: bit 0, clic per vettore completo')
        if self.selected is not None:
            self.inspect(self.selected)

    def _build(self):
        self.rebuild_count += 1
        self.canvas.delete('all')
        brain = self.brain
        sensors, nodes = brain['sensors'], brain['nodes']
        positions = [(90, 40+i*37) for i in range(sensors)]
        positions += [(290+(i%8)*115, 65+(i//8)*100) for i in range(len(nodes))]
        self.sensor_items = []
        self.node_items = []
        self.gate_labels = []
        for i, n in enumerate(nodes):
            x,y = positions[sensors+i]
            sources = [(n['a'],False)]
            if n['op'] not in ('COPY','NOT') or brain.get('learning'):
                sources.append((n['b'],False))
            if brain.get('learning'):
                rule=brain['learning'][i]
                sources += [(rule['eligibility'],True)] if rule['plastic'] else []
                sources += [(rule['hold'],True)] if rule['hold']>=0 else []
            for index,modulatory in sources:
                if 0<=index<len(positions):
                    px,py=positions[index]
                    self.canvas.create_line(px+18,py,x-18,y,arrow='last',
                                            fill='#c4b5fd' if modulatory else '#cbd5e1',
                                            dash=(4,3) if modulatory else ())
        for i in range(sensors):
            x,y=positions[i]
            item=self.canvas.create_rectangle(x-80,y-12,x+80,y+12,fill='#e2e8f0',outline='')
            self.sensor_items.append(item)
            self.canvas.create_text(x,y,text=f'S{i} {SENSORS[i] if i<len(SENSORS) else "input"}',font=('Segoe UI',8))
        for i,n in enumerate(nodes):
            x,y=positions[sensors+i]
            tag=f'neuron_{i}'
            item=self.canvas.create_oval(x-23,y-23,x+23,y+23,fill='#1e3a5f',
                                        outline='#f59e0b' if i in brain['outputs'] else '#64748b',width=3,tags=tag)
            self.node_items.append(item)
            self.canvas.create_text(x,y,text=f'N{i}',fill='white',tags=tag)
            self.gate_labels.append(self.canvas.create_text(x,y+35,text=n['op'],font=('Segoe UI',8),tags=tag))
            self.canvas.tag_bind(tag,'<Button-1>',lambda event,index=i:self.inspect(index))
        width=1220;height=max(sensors*37+70,math.ceil(len(nodes)/8)*100+100)
        self.canvas.create_text(250,height-25,anchor='w',text=f'Uscite bit 0,1,2: {brain["outputs"]} · Tratteggio: plasticità/hold; bordi oro: uscite')
        self.canvas.configure(scrollregion=(0,0,width,height))

    def inspect(self, index):
        if not self.brain or index>=len(self.brain['nodes']):
            return
        self.selected=index
        n=self.brain['nodes'][index];frame=self.frame
        previous=frame.get('previous_state',[])
        observation=frame.get('observation',[])
        inputs=observation+previous
        states=frame.get('neural_state',[])
        def value(source):return inputs[source] if source<len(inputs) else '?'
        rule=self.brain.get('learning',[])
        updates=[u for u in frame.get('plastic_updates',[]) if u['neuron']==index]
        text=(f'N{index}: operatore ereditato {n["op"]}; abilitato={n["active"]}; a={n["a"]} ({value(n["a"])}), '
              f'b={n["b"]} ({value(n["b"])})\n'
              f'Stato precedente={previous[index] if index<len(previous) else "?"}; '
              f'attuale={states[index] if index<len(states) else "?"}; '
              f'bit uscita={[i for i,v in enumerate(self.brain["outputs"]) if v==index]}\n')
        if self.brain.get('engine')==Limits.ENGINE:
            width=math.prod(self.brain['shape'])
            current=frame.get('vector_state',[])
            old=frame.get('vector_previous',[])
            text=f'Modulo {index} · forma {self.brain["shape"]} · {width} bit · abilitato={n["active"]}\n'
            text+=f'Ingressi vettoriali a={n["a"]}, b={n["b"]}; regola={self.brain["rules"][index]}\n'
            if index<len(current):
                text+=f'Precedente: {format(int(old[index],16),f"0{width}b") if index<len(old) else "?"}\n'
                text+=f'Attuale:    {format(int(current[index],16),f"0{width}b")}\n'
            else:text+='Stato vettoriale live non disponibile.\n'
            text+=f'Decodificatore RAM e contesto: {frame.get("decoder",{})}\n'
            text+=f'Memorie contestuali: {frame.get("contextual_entries",len(self.brain.get("contextual_memory",[])))}\n'
            text+=f'Scritture dopo feedback: {frame.get("vector_updates",[])}\n'
            text+=f'Errore previsione XOR: {frame.get("prediction_error")} (None = contesto sconosciuto)\n'
            text+='Il bit 0 nel grafo è una proiezione; gli output usano le corsie '+str(self.brain['output_lanes'])
            self.details.delete('1.0','end');self.details.insert('end',text)
            return
        if rule:
            tables=frame.get('plastic_tables',[])
            text+=f'Regola evoluta: {rule[index]}\nTabella dopo feedback: {tables[index] if index<len(tables) else "?"}; modifiche: {updates}\n'
        text+='* indica una funzione logica modificata dalla plasticità; il grafico mostra quella usata per la decisione.\n'
        text+='Gli ingressi si riferiscono alla decisione prima del movimento; gli stati attuali producono l’azione. Nessuna funzione cognitiva è inferita.'
        self.details.delete('1.0','end');self.details.insert('end',text)

    def compare(self):
        if not self.brain:return
        filename=filedialog.askopenfilename(filetypes=[('Cervello JSON','*.json')])
        if not filename:return
        try:
            with open(filename,encoding='utf-8') as stream:other=json.load(stream)
            nodes=other['nodes']
            changed=[i for i,(a,b) in enumerate(zip(self.brain['nodes'],nodes)) if
                     {k:a[k] for k in ('a','b','op','active')}!={k:b[k] for k in ('a','b','op','active')}]
            result=f'Confronto strutturale con {filename}\nNodi: {len(self.brain["nodes"])} / {len(nodes)}; nodi comuni diversi: {changed}\nUscite: {self.brain["outputs"]} / {other["outputs"]}\nRegole plastiche uguali: {self.brain.get("learning")==other.get("learning")}'
        except (OSError,ValueError,KeyError,TypeError) as exc:result=f'Confronto non disponibile: {exc}'
        self.selected=None
        self.details.delete('1.0','end');self.details.insert('end',result)
