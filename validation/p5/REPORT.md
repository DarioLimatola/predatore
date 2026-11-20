# P5 · Mondi, avversario neurale e trasferimento

Per allenare EVA sui sette scenari da una cartella di esperimento nuova:

```sh
python launcher.py --train --scenario-set extended --vault-root laboratorio --seed 83 --live
```

La suite predefinita `classic` mantiene i quattro scenari originali. Nel profilo `extended`, `maze_novel` costruisce un labirinto perfetto con corridoi, vicoli e un solo percorso a piedi fra due celle. Le pareti sono invalicabili anche con un salto. `delayed_reward` registra il primo contatto con la preda e conferma la cattura quattro decisioni dopo, con premio +20. Prima della conferma il premio è zero. Un tag a fine budget può quindi rimanere senza conferma. `intelligent_prey` istanzia una seconda rete booleana con lo stesso contratto di 15 sensori locali e otto azioni. La preda reagisce a collisioni, perdita di contatto visivo e cattura. Nelle normali run ciascun episodio parte dalla stessa configurazione iniziale dell'avversario. L'adattamento reciproco tra generazioni è un esperimento separato riproducibile.

Il replay include l'azione e il decoder della preda oppure i tick di contatto e di conferma, secondo il compito. Le coordinate e le distanze complete servono per misurare e visualizzare l'episodio. Entrambe le reti decidono dall'osservazione locale.

## Esperimento misurato

```sh
python -m tools.benchmark_worlds --output validation/p5/worlds.json
python -m unittest discover -s tests -p test_challenge_worlds.py -v
```

Seed 83, addestramento 8×8, tre generazioni. Per ciascuna coppia di candidati si eseguono cinque episodi identici di `intelligent_prey` con un massimo di 36 passi. Si confrontano incumbent contro incumbent, predatore mutato contro preda incumbent e predatore incumbent contro preda mutata. I predatori conservano quanto appreso nei cinque episodi, mentre le prede ereditano il circuito selezionato e ricominciano l'adattamento locale ad ogni episodio. Il confronto premia prima le catture del predatore, poi la loro rapidità. La preda ha l'obiettivo opposto. Tre candidati predatore sono rimasti al punteggio dell'incumbent. Una mutazione della preda è stata selezionata alla seconda generazione.

In valutazione ogni riga usa tre mappe inedite e quattro posizioni iniziali per mappa, per un totale di 12 episodi appaiati e un budget di 36 passi. Il cacciatore neurale riparte da una copia congelata in ciascun episodio. Di seguito le catture riuscite; il file JSON include seed, dati di episodio, intervalli Wilson descrittivi, collisioni e tempo di decisione del predatore neurale.

| Lato | Compito | Predatore iniziale | Dopo addestramento 8×8 | Random, 8 azioni | Greedy, sensori locali |
|---|---|---:|---:|---:|---:|
| 8 | Labirinto | 2 | 2 | 4 | 7 |
| 8 | Visibilità ridotta | 0 | 0 | 1 | 6 |
| 8 | Premio ritardato | 0 | 0 | 2 | 12 |
| 8 | Preda neurale | 3 | 3 | 4 | 12 |
| 12 | Labirinto | 0 | 0 | 3 | 4 |
| 12 | Visibilità ridotta | 0 | 0 | 1 | 2 |
| 12 | Premio ritardato | 0 | 0 | 0 | 7 |
| 12 | Preda neurale | 2 | 1 | 2 | 8 |

**Interpretazione.** Nella misura su 12×12 il predatore addestrato cattura la preda selezionata 1/12 volte. Riportando soltanto la preda al circuito iniziale le catture diventano 2/12. Tre mappe per condizione e un solo seed di addestramento lasciano ampia incertezza. La baseline Greedy a parità di sensori supera il predatore su tutte le condizioni osservate. Il gate di generalizzazione del predatore rimane aperto. Nel training ordinario rimane disponibile anche la baseline Greedy storica, etichettata come privilegiata perché legge la posizione vera della preda.

**Integrazione.** Run estesa su seed 87, lato 6, quattro esploratori, sette episodi per split, 16 passi e 32 neuroni. Al termine della prima generazione il monitor ha completato la lettura e i sette replay erano presenti. Ripartenza dal checkpoint con completamento della seconda generazione.
