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
        tools = ttk.Frame(self)
        tools.pack(fill='x',padx=10,pady=(0,5))
        ttk.Label(tools,text='Moduli').pack(side='left')
        self.group = tk.StringVar(value='Tutti')
        control=ttk.Combobox(tools,textvariable=self.group,values=('Tutti','Calcolati','Uscite','Credito aggiornato'),
                             state='readonly',width=18)
        control.pack(side='left',padx=(4,12))
        self.zoom=tk.StringVar(value='100%')
        ttk.Label(tools,text='Zoom').pack(side='left')
        zoom_control=ttk.Combobox(tools,textvariable=self.zoom,values=('60%','80%','100%','125%','150%'),
                                  state='readonly',width=6)
        zoom_control.pack(side='left',padx=(4,12))
        for selector in (control,zoom_control):
            selector.bind('<<ComboboxSelected>>',self.redraw)
        ttk.Button(tools, text='Confronta con cervello JSON…', command=self.compare).pack(side='left')
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

    def redraw(self, event=None):
        self.signature=None
        if self.brain:self.show(self.brain,self.frame,self.source)

    def show(self, brain, frame=None, source='Struttura salvata'):
        self.source=source
        if not brain:
            self.info.set(source + ' · stati neurali non disponibili')
            self.canvas.delete('all');self.signature=None;self.brain=None;self.frame={}
            self.selected=None;self.details.delete('1.0','end')
            return
        frame = frame or {}
        self.brain, self.frame = brain, frame
        signature = json.dumps({k:brain.get(k) for k in ('sensors','nodes','outputs','learning','shape','rules','output_lanes')},sort_keys=True)
        group=self.group.get() if hasattr(self,'group') else 'Tutti'
        if group=='Uscite':
            visible=set(brain['outputs'])
        elif group=='Calcolati':
            visible=set(frame.get('computed_nodes',()))
        elif group=='Credito aggiornato':
            visible={r['node'] for r in frame.get('relation_credit',{}).get('changed',())}
        else:visible=set(range(len(brain['nodes'])))
        self.visible=visible
        signature=(signature,group,tuple(sorted(visible)))
        if signature != self.signature:
            self.signature = signature
            self._build()
        states = frame.get('neural_state', [])
        observation = frame.get('observation', [])
        tables = [list(row) for row in frame.get('plastic_tables', [])]
        for update in frame.get('plastic_updates', []):
            if update['neuron'] < len(tables):tables[update['neuron']][update['entry']] = update['before']
        computed=set(frame.get('computed_nodes',range(len(brain['nodes']))))
        for i, item in self.node_items.items():
            n = brain['nodes'][i]
            color = '#94a3b8' if not n['active'] or i not in computed else ('#22c55e' if i<len(states) and states[i] else '#1e3a5f' if i<len(states) else '#64748b')
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
        if brain.get('engine') in Limits.SUPPORTED_ENGINES:
            width=frame.get('vector_width',math.prod(brain['shape']))
            for label in self.gate_labels.values():self.canvas.itemconfigure(label,text=f'LUT {width}b')
            self.info.set(f'{source} · passo {frame.get("t","—")} · {len(computed)}/{len(brain["nodes"])} moduli × {width} bit · azione {frame.get("action","—")} ({frame.get("decoder",{}).get("source","—")}) · memorie {frame.get("knowledge_entries",len(brain.get("knowledge",{}).get("rows",[])))} · verde/blu: bit 0 acceso/spento · grigio: inattivo o non misurato')
        credit=frame.get('relation_credit')
        if credit is not None:
            self.info.set(self.info.get()+f' · {credit["count"]} relazioni · traccia {credit["horizon"]} passi')
        if self.selected is not None:
            if self.selected in visible:self.inspect(self.selected)
            else:self.details.delete('1.0','end');self.selected=None

    def _build(self):
        self.rebuild_count += 1
        self.canvas.delete('all')
        brain = self.brain
        sensors, nodes = brain['sensors'], brain['nodes']
        ordered=sorted(self.visible)
        positions = {i:(90,40+i*37) for i in range(sensors)}
        positions.update({sensors+i:(290+(j%8)*115,65+(j//8)*100) for j,i in enumerate(ordered)})
        self.sensor_items = []
        self.node_items = {}
        self.gate_labels = {}
        for i in ordered:
            n=nodes[i]
            x,y = positions[sensors+i]
            sources = [(n['a'],False)]
            if brain.get('engine') in Limits.SUPPORTED_ENGINES:
                sources.append((n['b'],False))
                hold=brain['rules'][i]['hold']
                if hold>=0:sources.append((hold,True))
            elif n['op'] not in ('COPY','NOT') or brain.get('learning'):
                sources.append((n['b'],False))
            if brain.get('learning'):
                rule=brain['learning'][i]
                sources += [(rule['eligibility'],True)] if rule['plastic'] else []
                sources += [(rule['hold'],True)] if rule['hold']>=0 else []
            for index,modulatory in sources:
                if index in positions:
                    px,py=positions[index]
                    self.canvas.create_line(px+18,py,x-18,y,arrow='last',
                                            fill='#c4b5fd' if modulatory else '#cbd5e1',
                                            dash=(4,3) if modulatory else ())
        for i in range(sensors):
            x,y=positions[i]
            item=self.canvas.create_rectangle(x-80,y-12,x+80,y+12,fill='#e2e8f0',outline='')
            self.sensor_items.append(item)
            self.canvas.create_text(x,y,text=f'S{i} {SENSORS[i] if i<len(SENSORS) else "input"}',font=('Segoe UI',8))
        for i in ordered:
            n=nodes[i]
            x,y=positions[sensors+i]
            tag=f'neuron_{i}'
            item=self.canvas.create_oval(x-23,y-23,x+23,y+23,fill='#1e3a5f',
                                        outline='#f59e0b' if i in brain['outputs'] else '#64748b',width=3,tags=tag)
            self.node_items[i]=item
            self.canvas.create_text(x,y,text=f'N{i}',fill='white',tags=tag)
            self.gate_labels[i]=self.canvas.create_text(x,y+35,text=n['op'],font=('Segoe UI',8),tags=tag)
            self.canvas.tag_bind(tag,'<Button-1>',lambda event,index=i:self.inspect(index))
        width=1220;height=max(sensors*37+70,math.ceil(len(ordered)/8)*100+100)
        self.canvas.create_text(250,height-25,anchor='w',text=f'Moduli di uscita: {brain["outputs"]} · Tratteggio: hold; bordi oro: uscite')
        zoom=int(self.zoom.get().rstrip('%'))/100 if hasattr(self,'zoom') else 1
        if zoom!=1:self.canvas.scale('all',0,0,zoom,zoom)
        self.canvas.configure(scrollregion=(0,0,width*zoom,height*zoom))

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
        if self.brain.get('engine') in Limits.SUPPORTED_ENGINES:
            width=math.prod(self.brain['shape'])
            current=frame.get('vector_state',[])
            old=frame.get('vector_previous',[])
            text=f'Modulo {index} · forma {self.brain["rules"][index]["shape"]} · {width} bit · abilitato={n["active"]}\n'
            text+=f'Ingressi vettoriali a={n["a"]}, b={n["b"]}; regola={self.brain["rules"][index]}\n'
            if index<len(current):
                text+=f'Precedente: {format(int(old[index],16),f"0{width}b") if index<len(old) else "?"}\n'
                text+=f'Attuale:    {format(int(current[index],16),f"0{width}b")}\n'
            else:text+='Stato vettoriale live non disponibile.\n'
            if index in frame.get('computed_nodes',[]):
                text+='Stato calcolato dal circuito in questo passo.\n'
            elif 'computed_nodes' in frame:
                text+='Stato conservato; modulo non eseguito in questo passo.\n'
            text+=f'Maschera scritture LUT: {self.brain["rules"][index]["write"]:#x}\n'
            masks=frame.get('known_masks')
            if masks is not None and index<len(masks):text+=f'Maschera bit noti: {masks[index]}\n'
            credit=frame.get('relation_credit')
            if credit is not None:
                updates=[r for r in credit['changed'] if r['node']==index]
                text+=f'Credito relazionale · aggiornamenti registrati per il modulo: {len(updates)}'
                if credit.get('sample_truncated'):text+=' (campione parziale)'
                text+='\n'
                for row in updates:
                    text+=(f'  lane {row["lane"]} · stato {row["state"]} → bit azione {row["action_bit"]}: '
                           f'{row["before"]:+d} → {row["after"]:+d} ({row["width"]} bit)\n')
            text+=f'Decodificatore RAM e contesto: {frame.get("decoder",{})}\n'
            text+=f'Memorie contestuali: {frame.get("contextual_entries",len(self.brain.get("contextual_memory",{}).get("rows",[])))}\n'
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
            result=f'Confronto strutturale con {filename}\nNodi: {len(self.brain["nodes"])} / {len(nodes)}; nodi comuni diversi: {changed}\nUscite: {self.brain["outputs"]} / {other["outputs"]}\nForme: {self.brain.get("shape")} / {other.get("shape")}\nCorsie: {self.brain.get("output_lanes")} / {other.get("output_lanes")}\nRegole modificate: {[i for i,(a,b) in enumerate(zip(self.brain["rules"],other["rules"])) if a!=b]}'
        except (OSError,ValueError,KeyError,TypeError) as exc:result=f'Confronto non disponibile: {exc}'
        self.selected=None
        self.details.delete('1.0','end');self.details.insert('end',result)
