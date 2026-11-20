# P2 · Circuito LUT a 64 lane

La domanda sperimentale è se il motore nativo riproduce le azioni e riduce il costo totale della valutazione di episodi. Il riferimento immutabile è la revisione Predatore `8f724235accc041c9e11c700786ac55cec9e3123`. Il profilo Python di rilascio è in `validation/v605/profile.txt`. Il processo attraversa la frontiera Python/Rust una volta per episodio. Il circuito sperimentale usa `monade-core` 0.6.5, revisione esatta nel manifest e nel lock Cargo.

## Verifica funzionale

Il corpus `golden.json` contiene le mappe, i semi, le osservazioni registrate, la configurazione dei circuiti e i digest SHA-256 dei risultati V6.05. Copre Static, Moving, Maze e Hidden, 32/64/96 moduli, da uno a tre tick, memoria attiva/disattiva, encoder misto/broadcast e LUT/maschere di scrittura mutate. I digest coprono tutte le azioni e i vettori di tutti i moduli a ogni decisione, più posizioni, ricompense, collisioni e catture. Il test Python controlla il riferimento storico anche quando il compilatore manca.

Il processo Rust supporta il circuito congelato a 64 lane con rotazioni, scritture parziali, hold, sincronia ricorrente e selezione sparsa. Il programma Python rifiuta ritardi attivi, trasformazioni tensoriali non supportate, decoder associativi/plastici e credito strutturale. Il carico non produce aggiornamenti dei pesi. Non si afferma equivalenza per un training con memoria e plasticità attive.

Per ogni episodio, Rust riceve la sequenza delle osservazioni. Per i tre scenari con preda ferma esegue anche il circuito a ciclo chiuso. Python costruisce una tabella delle transizioni interrogando il vero World per ogni posizione raggiungibile, stato del sensore di collisione e azione. Una riga fornisce al circuito solo i bit dei sensori. L'azione seleziona la riga successiva. La verifica finale riapplica le azioni nel mondo e controlla traiettoria e score. Un eventuale suffisso dopo lo stallo viene scartato al medesimo passo della reference. Per Moving si usa lo stream registrato così che la prova disponga delle stesse osservazioni senza portare in Rust il generatore casuale Python.

## Misurazione

Il cronometro include costruzione tabella e serializzazione del circuito, avvio del processo, IPC, inferenza e scoring dell'episodio. La preparazione del genoma iniziale è comune ed esclusa. Si alterna l'ordine dei backend. Il rapporto è la mediana del tempo dell'intero gruppo Python divisa per la mediana dell'intero gruppo nativo. RSS del driver e massimo di un singolo figlio sono riportati separatamente: il picco del figlio può includere la memoria ereditata al fork. Le misure CPU/RSS si ottengono in processi freschi e includono anche la creazione dei controller. Windows non espone tutti i contatori RUSAGE.

Due prove esplorative da nove repliche hanno dato risultati ai due lati della parità sul tempo di parete. Per la successiva misura registrata si usano 31 repliche senza eliminare gli outlier. La promozione richiederebbe almeno il 10% di risparmio wall, parità funzionale, e conferma in un run completo di training comprensivo di salvataggi. Il risultato della prova circuit-only non autorizza da solo la promozione. Il rollback consiste nell'usare l'entry point ordinario Python, che è già quello predefinito.

## Riproduzione

Dalla directory principale, con Rust e Cargo 1.97 e Python 3.10+:

```sh
cargo test --manifest-path native/Cargo.toml --locked
cargo build --release --manifest-path native/Cargo.toml --locked
python tools/benchmark_native.py --repetitions 31 --output validation/p2/benchmark-repeat.json
```

Cargo richiede accesso iniziale a GitHub e crates.io, oppure una cache dei sorgenti. Linux e Windows usano lo stesso script; su Windows identifica automaticamente il suffisso .exe.
