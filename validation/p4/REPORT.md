# EVA, trasferimento e protezione della memoria

## Risultato

Nei nove casi storici abbiamo riprodotto i nove tassi finali di cattura su holdout e tutte le 62 regressioni dopo apprendimento personale, sulle 90 generazioni. In 34 di questi 62 casi EVA perde almeno una cattura di validazione. Negli altri 28 la classifica peggiora a parità di catture. Il risultato dipende anche da stallo e ricompensa, che la V6.05 usa per ordinare gli agenti a parità di catture.

Nella prova fattoriale breve la protezione di EVA migliora la seconda run della linea 42 da 1 a 8 catture su 32. Nelle linee 117 e 809 il risultato resta rispettivamente 7/32 e 5/32. Anche la selezione degli esploratori per apprendimento ottiene gli stessi totali. Tre linee e due generazioni per run sono insufficienti per promuovere automaticamente una delle varianti.

## Regola applicata

Con `--max-scenario-loss N`, una proposta deve superare o eguagliare il ranking bilanciato e rispettare la perdita massima di N catture rispetto allo stato precedente, su ciascuna coppia di task e scenario. Il ranking bilancia la run corrente con le run ricordate. L'aggiornamento personale è valutato dopo la simulazione sulle stesse mappe di validazione, con copie adattive isolate. Il controllo sullo stato a inizio generazione evita che un trasferimento e il successivo training personale consumino due volte la tolleranza. Se il candidato personale scende sotto il ranking precedente o supera la tolleranza, il motore ripristina lo stato stabile. La soglia è espressa in catture per lo specifico budget di validazione.

Il checkpoint SMITH conserva il candidato personale, lo snapshot stabile precedente e i rispettivi identificatori in `eva_guard`. Le metriche di training rimangono quelle dell'episodio eseguito. I contatori in `learning_channels` distinguono aggiornamenti personali, distillazione, adozione di un agente e interventi sul circuito. Il file `evolution/consolidation.jsonl` registra candidati, fonte, tempi e differenze appaiate per scenario. Nel modo protetto i campioni provenienti dal medesimo snapshot iniziale di teacher concorrono con un solo voto per contesto. È una scelta conservativa. Due individui con la stessa rete iniziale possono avere esperienze distinte.

Con `--select-for-learning`, ogni esploratore affronta una mappa-prova di training prima e dopo i propri episodi. I due test usano copie congelate, mappe e semi uguali tra loro e diversi dagli episodi che forniscono feedback. La popolazione viene ordinata per catture finali più incremento delle catture. A parità si usano le catture finali della prova e il ranking storico. Elite, specialisti, novelty e lineage continuano a partecipare alla selezione. SMITH usa la prestazione di validazione dello stato persistente effettivamente conservato. Le quattro procedure che può selezionare restano esplicite.

## Esperimento di retention e trasferimento

Prima del confronto abbiamo fissato i bracci, la soglia zero e il budget. Tre linee indipendenti, seed iniziali 42, 117 e 809, seguiti rispettivamente da 43, 118 e 810. Ogni run usa quattro esploratori, due generazioni, 32 moduli, mappe 8×8 e un massimo di 32 passi. Per generazione e agente ci sono otto episodi di training. Ogni validazione comprende dodici episodi, tre per scenario. Ogni holdout ne comprende 32, otto per scenario. Dopo la seconda run rivalutiamo anche le mappe e i semi di holdout della prima. Le coppie prima/dopo usano copie adattive e la stessa sequenza. I risultati di holdout vengono esclusi da tutte le decisioni.

| Guardia | Selezione per apprendimento | Prima run, catture su 96 | Seconda run, catture su 32 per linea (42, 117, 809) | Prima run rivalutata dopo la seconda, su 96 | Rollback personali nelle 12 generazioni | Secondi complessivi per sei run |
|---|---|---:|---|---:|---:|---:|
| No | No | 6 | 1, 7, 5 | 6 | 0 | 12,9 |
| Sì | No | 9 | 8, 7, 5 | 9 | 5 | 12,3 |
| No | Sì | 9 | 8, 7, 5 | 9 | 0 | 13,4 |
| Sì | Sì | 9 | 8, 7, 5 | 9 | 5 | 14,0 |

In questo campione le catture di retention restano uguali anche suddividendo per scenario. La riproduzione del miglioramento in una sola linea e il numero ridotto di eventi positivi lasciano aperta la generalizzazione. La condizione con guardia comprende anche il raggruppamento delle fonti di distillazione per identità iniziale. I tempi sono osservazioni descrittive su macchina condivisa. I dati per seed, scenario, conteggio, passi delle catture, stallo, reward, baseline, tempo CPU e wall sono in `paired.json`.

## Diagnosi delle regressioni V6.05

Tre seed (42, 117, 809), tre dimensioni fisse (32, 64, 96), sei esploratori, dieci generazioni, mappe 8×8, otto episodi di training e otto di validazione per confronto, 32 di holdout per run, 60 passi. Abbiamo replicato risultati e budget da `validation/v605/controlled.json`. A ogni generazione abbiamo affiancato al training reale copie usa e getta. Una riceve l'aggiornamento di feedback a transizioni alterne. Due ulteriori sonde disattivano il decoder associativo durante la valutazione prima e dopo il training personale. Quando cambia una scelta dell'agente, anche la traiettoria può divergere.

| Braccio dopo il training personale | Catture su 720 episodi di validazione |
|---|---:|
| Feedback originale | 315 |
| Scritture di feedback a transizioni alterne | 325 |
| Decoder associativo disattivato in valutazione | 116 |

Nel sottoinsieme delle 62 regressioni, il braccio a transizioni alterne supera il candidato originale nel ranking bilanciato 44 volte. Nelle 90 generazioni complessive succede 50 volte. L'ablazione del decoder elimina in queste 62 generazioni tutte le perdite di cattura *prima/dopo* osservate con il decoder attivo, ma la prestazione assoluta del braccio senza decoder è molto più bassa. Questo risultato indica che la lettura della memoria partecipa sia alla capacità di cattura sia alla sensibilità agli aggiornamenti. Una riduzione globale permanente degli update richiede nuovi seed e una verifica sulle sequenze più lunghe. Le generazioni e gli episodi consecutivi della stessa linea sono correlati. Il ruolo causale della scelta di ogni singolo teacher resta aperto.

`diagnosis.json` contiene i cinque bracci, i dettagli per scenario, il teacher selezionato e la chiave di confronto di ciascuna generazione. Le due procedure si riproducono dalla directory del progetto.

```bash
python tools/benchmark_eva_guard.py
python tools/diagnose_eva_regression.py
python -m unittest discover -s tests -p test_eva_guard.py
```
