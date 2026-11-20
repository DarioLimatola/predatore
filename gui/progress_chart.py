"""One palette and legend for every generation, scenario and metric."""
from engine.settings import Limits

EVA_COLOR = '#2563eb'
POPULATION_COLOR = '#16a34a'
VALIDATION_COLOR = '#d97706'
BEST_COLOR = '#8b5cf6'
BASELINE_COLORS = {'random': '#64748b', 'greedy': '#db2777', 'sensor_greedy':'#0891b2'}


def draw_progress(canvas, history, metric, validation=(), baselines=None):
    canvas.delete('all')
    w=max(500,canvas.winfo_width()); h=max(320,canvas.winfo_height())
    canvas.create_text(65,25,text=metric,anchor='w',fill='#14263e')
    legend = 'Blu: '+Limits.EVA_LABEL+' · Verde: Popolazione — media'
    series=[]
    if metric=='Catture':
        series=[([(r['generation'],r.get(k)) for r in history],c) for k,c in
                [('capture_rate',EVA_COLOR),('population_mean_capture',POPULATION_COLOR)]]
        if validation:
            series.append(([(r['generation'],r['validation'].get('capture_rate')) for r in validation],VALIDATION_COLOR))
            legend+=' · Arancio: EVA — validazione'
        for name,color in BASELINE_COLORS.items():
            value=(baselines or {}).get('train',{}).get(name,{}).get('capture_rate')
            if value is not None and history:
                series.append(([(history[0]['generation'],value),(history[-1]['generation'],value)],color))
                legend+={'random':' · Grigio: casuale','greedy':' · Magenta: inseguitore privilegiato','sensor_greedy':' · Turchese: inseguitore con sensori'}[name]
        peak=1
    elif metric=='Passi per cattura':
        series=[([(r['generation'],r.get(k)) for r in history],c) for k,c in
                [('successful_mean_steps',EVA_COLOR),('population_successful_mean_steps',POPULATION_COLOR),('best_capture_steps',BEST_COLOR)]]
        legend+=' · Viola: migliore cattura di EVA · Meno passi = meglio'
        peak=max([v for points,_ in series for _,v in points if v is not None]+[1])
    else:
        series=[([(r['generation'],r.get('active_neurons')) for r in history],EVA_COLOR),
                ([(r['generation'],round(r['population_mean_neurons']) if r.get('population_mean_neurons') is not None else None) for r in history],POPULATION_COLOR)]
        legend+=' arrotondata · Valori in neuroni'
        peak=max([v for points,_ in series for _,v in points if v is not None]+[1])
    if not history:
        canvas.create_text(w/2,h/2,text='Nessuna generazione disponibile')
        return legend
    lo=min(r['generation'] for r in history); hi=max(r['generation'] for r in history)
    def point(g,v):return 65+(g-lo)*(w-100)/max(1,hi-lo),h-55-v*(h-120)/peak
    neuron_metric=metric not in ('Catture','Passi per cattura')
    ticks=sorted({round(peak*i/4) for i in range(5)}) if neuron_metric else [peak*i/4 for i in range(5)]
    for value in ticks:
        x,y=point(lo,value)
        canvas.create_line(65,y,w-35,y,fill='#e2e8f0')
        label=f'{value:.0%}' if metric=='Catture' else str(int(value)) if neuron_metric else f'{value:.1f}'
        canvas.create_text(34,y,text=label)
    for g in sorted({round(lo+(hi-lo)*i/4) for i in range(5)}):
        x,y=point(g,0);canvas.create_text(x,y+16,text=str(g))
    canvas.create_text(w/2,h-12,text='Generazione')
    for points,color in series:
        previous=None
        for g,v in points:
            if v is None:previous=None;continue
            x,y=point(g,v)
            if previous:canvas.create_line(*previous,x,y,fill=color,width=2)
            else:canvas.create_oval(x-2,y-2,x+2,y+2,fill=color,outline=color)
            previous=(x,y)
    return legend
