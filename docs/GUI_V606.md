# Laboratorio evolutivo V6.06

Aprire `python launcher.py --gui` dalla cartella del progetto. La casella **Esperimento** seleziona una run di EvolutionVault per tutte le viste. **Esperimento** raccoglie avvio, ripresa, arresto a fine generazione, log e impostazioni. Le nuove run usano il curriculum classico e il credito relazionale disattivato. La selezione *Estesi* aggiunge labirinto a corridoi, ricompensa ritardata e preda neurale. Gli altri controlli sperimentali attivano rispettivamente la traccia, il limite alle perdite di catture su uno scenario e la selezione per capacità di apprendere.

## Arena e replay

| Vista | Sorgente e controlli |
|---|---|
| Training reale | Il worker pubblica fotogrammi campionati. Il filtro seleziona slot e scenario. Pausa e passo scorrono il buffer. Il mosaico conserva l'ultima osservazione di ciascuno slot, anche se gli episodi sono avvenuti in momenti diversi. |
| Simulazione agenti | EVA, esploratori e controlli agiscono in mondi indipendenti con la stessa mappa e lo stesso seed iniziale. Selezionare il confronto con sensori equivalenti o l'inseguitore privilegiato che conosce la posizione della preda. |
| Replay registrato | Selezionare l'episodio e trascinare la linea temporale per mettere in pausa e ispezionare un fotogramma. Il cervello visualizzato è quello della registrazione o uno snapshot con lo stesso identificatore. |

L'arena mostra la posizione vera della preda all'osservatore. La visibilità riportata accanto all'azione viene dall'osservazione precedente al movimento. Il pannello riporta catture, ricompensa differita o azione della preda quando il fotogramma contiene tali dati. Il tempo dell'episodio è espresso in passi. Intervalli di campionamento e pacchetti persi possono produrre salti nel live.

## Risultati, cervello, archivio

**Risultati** riunisce l'evoluzione per generazione e i benchmark per split, agente e scenario. Nei nuovi record la tabella mostra successi/tentativi, intervalli di Wilson per scenario, media e mediana dei passi per le catture riuscite. Il trattino indica una misura assente, anche per run storiche. L'intervallo descrive gli episodi campionati su quel compito. La varianza fra mondi e run richiede prove separate.

**Cervello e memoria** usa il medesimo fotogramma dell'arena. In preview si può scegliere l'esploratore. I filtri restringono il grafo ai moduli calcolati, alle uscite o ai moduli per cui il campione contiene aggiornamenti di credito. Selezionare un nodo per LUT, forma, maschera di scrittura, stato osservato e precedente, decisione RAM e variazioni dei contatori per relazione. La maschera di scrittura descrive la LUT. Le maschere dei bit noti vengono visualizzate solo quando il frame le fornisce. La vista contrassegna i campioni troncati. Lo zoom regola la distanza fra i nodi; le etichette mantengono la propria dimensione.

**Archivio** apre SMITH e diagnostica per la run selezionata. Lo stato SMITH proviene dal checkpoint di quella run. Quando sono disponibili le evidenze V6.06, sono visibili il candidato personale, il ripristino eventuale e i conteggi dei canali di apprendimento.

## Riproduzione delle misure

Eseguire dalla radice con Python e Tk disponibili. L'ultimo comando richiede un display nativo o virtuale.

```sh
python -m unittest discover -s tests -p test_lab_sources.py -v
python tools/benchmark_telemetry.py --out validation/p6/telemetry.json
python tools/benchmark_live_window.py --out validation/p6/live_window.json
```

Il benchmark della telemetria confronta dodici addestramenti su sei seed, con alternanza dell'ordine, stesso budget e profili iniziali nuovi. L'esperimento include salvataggi, diagnostica finale e chiusura del writer. I genomi finali e le catture holdout coincidono in tutte le coppie. Il costo mediano osservato è +2,28%, con un intervallo fra i singoli seed da −2,41% a +12,97%. Il confronto con finestra Tk visibile include il rendering e il polling della GUI in parallelo al worker. Su quattro seed distinti, lo stesso genoma e le stesse catture sono stati osservati nelle quattro coppie. La variazione mediana con finestra aperta è +2,32%, da −1,45% a +10,62%. La misura include il percorso di polling precedente all’ultima ottimizzazione di cache dei file invariati. I JSON conservano dati grezzi, ambiente e protocolli. Il budget mediano fissato per questi due confronti era il 10%.

I test nativi usano Tk su Linux con Xorg dummy. Esercitano anche il caricamento del replay esteso, lo scorrimento temporale, la preda neurale, il filtro per uscite, i valori relazionali e lo zoom. Il test di file bloccato inietta un errore di accesso e controlla che il polling riprenda dopo il ripristino del file. La prova su un desktop Windows reale resta aperta.
