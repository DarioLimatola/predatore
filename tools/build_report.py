"""Build the release validation report from measured JSON and unit-test logs."""
import json
from pathlib import Path
import re
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'validation/v605'


def read(name):
    return json.loads((DATA/name).read_text(encoding='utf-8'))


def main():
    rows=read('controlled.json');lines=['# Rapporto di verifica V6.05','',
        '## Protocollo','']
    config=rows[0]['config']
    assert len(rows)==9
    assert {r['config']['seed'] for r in rows}=={42,117,809}
    lines += [
        f"Tre seed e tre budget fissi: 32, 64, 96 moduli da 64 bit. Per confronto gli altri parametri sono identici: {config['population']} esploratori, {config['generations']} generazioni, {config['episodes']} episodi di training per generazione, {config['validation_episodes']} episodi di validazione, {config['holdout_episodes']} episodi di holdout, arene 2D {config['size']}×{config['size']} e {config['steps']} passi massimi per episodio.",
        '',
        "L'holdout confronta EVA iniziale, EVA finale e popolazione finale con copie adattive su stessi episodi. Nessun feedback o seed di holdout entra nella selezione. I processi sono eseguiti in sequenza, con memoria misurata isolatamente nei nove casi del confronto.",
        '',"## Confronto dei moduli",'',
        '| Moduli | EVA iniziale | EVA finale | Pop. finale | Tempo | Picco RAM |',
        '| --- | --- | --- | --- | --- | --- |']
    pct=lambda n:f'{n*100:.2f}%'.replace('.',',')
    summary={}
    for n in (32,64,96):
        cases=[row for row in rows if row['config']['initial_neurons']==n]
        avg=lambda k:mean(r[k] for r in cases)
        memory=[r['peak_mib'] for r in cases if r['peak_mib'] is not None]
        ram=f'{mean(memory):.1f} MiB' if memory else 'n.d.'
        summary[n]=avg('final')
        lines.append(f"| {n} | {pct(avg('initial'))} | {pct(avg('final'))} | {pct(avg('population'))} | {avg('seconds'):.1f} s | {ram} |")
    best=max(summary,key=summary.get)
    regressions=sum(r['final']<r['initial'] for r in rows)
    below=sum(r['final']<r['population'] for r in rows)
    personal=sum(r['personal_regressions'] for r in rows)
    lines+=['',
        f'{best} moduli hanno la media EVA finale più alta in questo campione. EVA regredisce in {regressions}/9 casi e chiude sotto la media degli esploratori in {below}/9. Le regressioni di validazione dopo il training personale sono {personal}/{9*config["generations"]} generazioni. Tre seed non bastano per scegliere una dimensione generale né per cambiare i default 32/64/128.',
        '',
        '| Moduli | Seed | EVA iniziale | EVA finale | Pop. finale | Eseguiti EVA |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        c=r['config']
        lines.append(f"| {c['initial_neurons']} | {c['seed']} | {pct(r['initial'])} | {pct(r['final'])} | {pct(r['population'])} | {r['executed']} |")
    lines+=['','## Continuità tra run','',
            '| Linea | Seed | EVA iniziale | EVA finale | Differenza |',
            '| --- | --- | --- | --- | --- |']
    for first,second in read('continuity.json'):
        assert first['final_id']==second['initial_id']
        for r in (first,second):
            delta=f"{100*(r['final']-r['initial']):+.2f} pp".replace('.',',')
            lines.append(f"| {first['config']['seed']} | {r['config']['seed']} | {pct(r['initial'])} | {pct(r['final'])} | {delta} |")
    lines+=['',"L'identità dello snapshot finale della prima run coincide con quella iniziale della seconda in entrambe le linee. La prestazione iniziale/finale si confronta entro ogni run; seed diversi cambiano il test. La persistenza è stata verificata, la crescita monotona no.",'',
        '## Prestazioni','',
        '| Moduli | Eseguiti | Completo | Sparso | Rapporto |',
        '| --- | --- | --- | --- | --- |']
    for r in read('inference.json'):
        lines.append(f"| {r['modules']} | {r['executed']} | {r['full']*1000:.1f} ms | {r['sparse']*1000:.1f} ms | {r['speedup']:.2f}× |")
    profile=(DATA/'profile.txt').read_text(encoding='utf-8')
    def cumulative(function):
        for line in profile.splitlines():
            if line.endswith('('+function+')'):
                return float(line.split()[3])
        raise ValueError(function)
    decide,rollout,write=(cumulative(k) for k in ('decide','rollout','atomic_json'))
    trials=sum(sum(r['decoder_counts'].values()) for r in rows)
    hdc=sum(r['decoder_counts']['hdc'] for r in rows)
    lines+=['',
        "Mediana di tre ripetizioni per 1500 decisioni identiche (seed del circuito 42, decoder di memoria disattivato, senza feedback). Le azioni coincidono tra esecuzione completa e sparsa. Il numero di moduli eseguiti dipende dai collegamenti e non cresce necessariamente con il totale.",
        '',
        f"Profilo cProfile separato: {rollout:.2f} s cumulativi in rollout, {decide:.2f} s in decide e {write:.2f} s in atomic_json. Sono tempi inclusivi e non si sommano. La simulazione, la validazione e la serializzazione incidono sul tempo complessivo; accelerare una singola decisione non produce lo stesso guadagno su tutta la run.",
        '',
        f"Nei nove training la decodifica finale HDC compare {hdc} volte su {trials} decisioni di EVA. La frequenza misura l'uso, non il beneficio causale. Le ablation nelle run tengono fissi i seed ma non garantiscono generalizzazione.",
        '',
        '## Test e riproduzione','']
    test_log=(DATA/'tests.txt').read_text(encoding='utf-8')
    count=int(re.search(r'Ran (\d+) tests',test_log)[1])
    skipped=int(re.search(r'OK \(skipped=(\d+)\)',test_log)[1])
    lines += [f'{count-skipped} test riusciti; {skipped} saltato (Tk nativo richiede un display). Verificati callback GUI, ripresa deterministica, persistenza tra run, memoria, distillazione e corrispondenza del calcolo sparso con quello completo. Esempio XOR: 50% prima del feedback e 100% sui quattro casi dopo il training.',
        '',
        '    python -m unittest discover -s tests -v',
        '    python tools/benchmark_release.py',
        '    python examples/binary_task.py',
        '',
        'Risultati individuali e configurazioni: controlled.json, continuity.json, inference.json, environment.json, profile.txt e profile_run.json. Le misure si riferiscono al sorgente ricostruito e verificato in questa consegna. Il benchmark include il costo di salvataggio, valutazioni e diagnostica.']
    (DATA/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
