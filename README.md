# PREDATORE V6.04 — EVA e SMITH

Estrarre lo ZIP in una cartella nuova. Dentro l’unica cartella `predatore`,
aprire `AVVIA_PREDATORE_V6.bat` su Windows oppure eseguire
`python3 launcher.py --gui`. Serve Python 3.10+ con Tkinter; nessuna libreria
numerica esterna è richiesta.

## Ruoli

| Componente | Funzione | Stato conservato |
|---|---|---|
| Esploratori | Imparano nell’ambiente e producono discendenti mutati | Circuito, RAM e memoria contestuale ereditati; una piccola coorte sopravvive tra run |
| **EVA - learning agent** | Impara dai migliori esploratori di ogni generazione e nei propri episodi | Un solo circuito con le sue memorie operative, salvato a ogni generazione |
| **SMITH - agent supervisor** | Conserva l’esperienza collettiva, seleziona i trasferimenti e apprende quali strategie utilizzare | Archivio di esperienze, specialisti, compiti di riferimento, statistiche e riepiloghi delle run; custodisce anche lo stato di EVA |

In Addestramento si configura il **profilo SMITH**. Lo stesso nome riprende la
stessa linea di memoria e la stessa EVA tra run. Un nome nuovo crea una linea
indipendente. Per una nuova run cambiare seed; Riprendi checkpoint continua
la configurazione salvata e permette di estendere le generazioni.

La popolazione contiene N esploratori; EVA è un agente aggiuntivo. SMITH è il
supervisore e non appare come un predatore nell’Arena. Il pannello SMITH e i
risultati di EVA seguono l’esperimento selezionato, anche quando appartiene a
un altro profilo.

## Apprendimento di ogni generazione

1. Gli esploratori eseguono gli episodi e aggiornano le proprie memorie.
2. SMITH seleziona il leader e gli specialisti per scenario, fino a tre insegnanti.
   Conserva campioni di tutti gli agenti, ma insegna a EVA usando i migliori
   correnti e le esperienze dei precedenti insegnanti ancora nell’archivio.
3. Il trasferimento confronta una distillazione sul circuito di EVA, una sul
   miglior insegnante, i circuiti intatti degli insegnanti e modifiche locali
   guidate dal feedback. Una copia identica viene valutata una sola volta.
4. SMITH sceglie la proposta con il punteggio di validazione migliore rispetto
   all’EVA corrente. Le prove sono copie isolate; le loro nuove esperienze non
   vengono trasferite. La selezione può anche mantenere il circuito corrente.
5. **EVA esegue sempre i propri episodi con apprendimento attivo.** Le modifiche
   vengono salvate anche quando nessuna proposta di trasferimento è selezionata.
6. Gli esploratori migliori diventano élite; gli altri discendenti ereditano
   memorie e subiscono mutazioni funzionali e strutturali.

Il punteggio di selezione dà metà del peso alle catture sui compiti della run
corrente e metà alla media dei compiti passati ricordati. Con una sola run usa
soltanto quella. I successivi criteri di spareggio sono catture correnti, meno
stalli, reward e meno neuroni. Catture e stalli sono confrontati con rapporti
razionali esatti; i log mostrano anche i guadagni e le perdite per scenario.

È stato eliminato il veto su qualsiasi minima regressione di ogni vecchio
scenario. Una perdita locale può essere compensata da un guadagno maggiore.
Questo permette trasferimenti prima bloccati, ma ammette compromessi e
possibile dimenticanza. La selezione è empirica: non garantisce una crescita
monotona né prestazioni sempre superiori alla popolazione.

## Neuroni e risorse

Predefiniti: **minimo 32, base 64, massimo 128**. Ogni valore ha un toggle,
uno slider e un campo numerico intero. Quando il toggle è spento viene usato il
valore predefinito. Vale sempre `32 <= minimo <= base <= massimo <= 4096`;
4096 è il limite tecnico personalizzabile, non il massimo predefinito.

La base inizializza le nuove reti. Nei discendenti mutati, una scelta simmetrica
può aggiungere un neurone, rimuoverne uno oppure mantenere la dimensione,
rispettando i limiti. Le rimozioni eliminano fisicamente il modulo e rimappano
connessioni, gate e uscite. Le aggiunte si collegano a un’uscita per poter
influenzare il comportamento. La selezione valuta le conseguenze; a prestazioni
uguali preferisce meno neuroni. EVA cambia struttura attraverso il trasferimento
selezionato. Non esistono intervalli di crescita/potatura né un obbligo di
raggiungere il massimo.

Su una linea già esistente la base non reinizializza EVA. Cambiare minimo o
massimo in una nuova run adatta i circuiti ai nuovi limiti; ciò può modificare il
comportamento, mentre le memorie di azione restano. La ripresa dello stesso
checkpoint mantiene i limiti salvati.

La forma predefinita `8,8` indica 64 bit per modulo, distinti dal numero di neuroni.
Gli assi sono ancora metadati; non sono operatori spaziali appresi. `Memorie RAM`
limita ciascuna delle tre tabelle: RAM breve, predittore e RAM contestuale.
`Esperienze condivise` limita le transizioni nell’archivio SMITH. Entrambi i
budget predefiniti valgono 2048. Le memorie temporanee si azzerano per episodio,
quelle apprese persistono. I budget centrali sono in `engine/settings.py`.

## GUI e misure

- **Addestramento / Evoluzione:** blu = EVA - learning agent; verde = media degli
  esploratori. Entrambi apprendono sugli stessi scenari, seed, episodi e limiti
  di passi. Il risultato è quello degli episodi realmente eseguiti.
- **Validazione:** arancio, quando presente. Rimane distinta dal training e
  riguarda copie isolate di EVA. Il test finale non guida le selezioni.
- **Neuroni:** valori interi; la media della popolazione è mostrata arrotondata.
- **Arena live:** slot 0…N−1 = esploratori, slot N = EVA. N è la popolazione
  configurata. L’osservazione non cambia l’apprendimento.
- **Anteprima:** simulazioni indipendenti dei cervelli della generazione
  selezionata; non scrive le esperienze nel profilo.
- **Replay:** episodi registrati. Brain Inspector, diagnostica e Vault restano
  disponibili. Un unico file `brains/eva.json` identifica EVA durante e dopo la run.

Validazione e test usano copie con apprendimento locale attivo e scartano le
modifiche alla fine della prova. Non sono altri agenti persistenti. Il benchmark
finale applica lo stesso protocollo a EVA e a tutti gli esploratori. Pertanto i
valori della V6.04 non sono direttamente confrontabili con i vecchi test che
misuravano le reti senza apprendimento.

## Memoria e limiti

Il contesto booleano contiene osservazione corrente e tre coppie precedenti
osservazione/azione: 72 bit con i sensori attuali. I contatori positivi e negativi
per le otto azioni usano piani di bit saturanti. La distillazione aggrega voti per
insegnante; ripassare lo stesso archivio non inventa nuovi voti indipendenti.

SMITH conserva campioni, non tutte le transizioni per sempre. Conserva fino a
quattro esploratori tra run, otto compiti di riferimento, 64 riepiloghi e
statistiche cumulative delle due strategie `consolidate` / `explore`.
Questa metaconoscenza riguarda il training del laboratorio. La distillazione
può perdere informazioni e subire interferenze; più neuroni non garantiscono
maggiore intelligenza. Vedere `VECTOR_PROTOCOL.md` e `validation/v604/REPORT.md`.

## Comandi

```bash
python launcher.py --train --seed 42 --generations 100
python launcher.py --train --seed 43 --base-neurons 64 --min-neurons 32 --max-neurons 128
python launcher.py --train --resume --seed 43 --size 8 --generations 200
python launcher.py --smith-status --smith-profile SMITH
python -m unittest discover -s tests -v
python tools/benchmark_learning.py
python tools/package_release.py
```

`--no-context-distillation` e `--no-circuit-plasticity` servono a esperimenti
controllati sullo stesso motore. Nessun motore alternativo o importatore delle
altre versioni è incluso. Formati: `boolean-vector/4`, `smith.supervisor/1`.
Estrarre in una cartella nuova; non sovrapporre il precedente EvolutionVault.
