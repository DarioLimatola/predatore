# Verifica V6.04

## Esito funzionale

`python -m unittest discover -s tests -v`: 96 esecuzioni, 95 superate e una
saltata (test nativo Tkinter: nessun display disponibile). Il log completo è
`tests.txt`; alcune classi riusano gli stessi contratti di callback.
I test con widget simulati verificano eventi, filtri, controlli e dati; non
sostituiscono una verifica visiva nativa su Windows.

Copertura rilevante:

- Catture EVA ricostruite dai frame degli episodi reali, uguali al dato del
  training; stessi seed, scenari ed episodi per ogni esploratore e per EVA.
- Apprendimento e salvataggio di EVA anche con tutti i trasferimenti respinti.
- Trasferimento che accetta un guadagno complessivo con perdita locale e registra
  tale perdita; rifiuto di candidati con punteggio bilanciato inferiore.
- Selezione di un insegnante più piccolo a prestazioni uguali, senza timer.
- 300 mutazioni con rimozione/rimappatura di nodi, riferimenti ricorrenti, gate e
  uscite; rispetto dei limiti e caricamento delle reti risultanti.
- Memoria contestuale, isolamento delle prove, confini fra training e test,
  continuità tra run e persistenza della memoria SMITH.
- Ripresa identica all’esecuzione ininterrotta, recupero dopo interruzione della
  finalizzazione, lock e gestione dei checkpoint bloccati su Windows.
- Tre slider/toggle e argomenti CLI, rifiuto di limiti incoerenti, cambio run e
  profilo, palette e assi interi, anteprima/replay/inspector, assenza di effetto
  del monitor e dell’osservatore sui risultati.

## Prove su più run

Tre linee indipendenti; due run consecutive per linea: 801→802, 811→812,
821→822. Ogni run: 6 generazioni, 8 esploratori più EVA, 16 episodi di training,
16 di validazione, 80 di test finale, 80 passi, griglia 8×8; base/min/max
64/32/128, 256 righe per memoria e 256 eventi condivisi. Quattro scenari.
Le configurazioni esatte e i conteggi sono in `benchmark.json`; riproducibili
con `python tools/benchmark_learning.py`.

Il confronto iniziale/finale usa la stessa suite riservata con copie adattive,
come il confronto EVA/popolazione. Le prove sono state eseguite dopo aver fissato
l’algoritmo e non sono state usate per scegliere nuove regole. Non è un’ablazione
né un confronto causale con la V6.03, che usava un protocollo diverso.

| Seed | Training EVA | Training popolazione | Test EVA iniziale | Test EVA finale | Test popolazione | Neuroni EVA |
|---|---:|---:|---:|---:|---:|---:|
| 801 | 62.50% | 45.31% | 42.50% | 26.25% | 34.22% | 64 |
| 802 | 62.50% | 42.97% | 48.75% | 75.00% | 50.00% | 65 |
| 811 | 37.50% | 39.06% | 43.75% | 45.00% | 42.03% | 62 |
| 812 | 25.00% | 42.97% | 40.00% | 38.75% | 37.66% | 62 |
| 821 | 31.25% | 39.84% | 31.25% | 33.75% | 32.81% | 64 |
| 822 | 43.75% | 34.38% | 31.25% | 33.75% | 32.66% | 64 |

Sono osservati 36 aggiornamenti personali di EVA su 36 generazioni, 34
trasferimenti selezionati e 806 scritture contestuali nelle candidate scelte.
Queste sono scritture, non 806 conoscenze indipendenti o contesti nuovi distinti.
La seconda run di ciascuna linea eredita esattamente circuito e conoscenze della
prima (confronto degli snapshot, escluso lo stato temporaneo), e SMITH ricorda
entrambe le run. Le dimensioni esplorate includono reti sia più piccole sia più
grandi della base: 59–68 neuroni in queste prove brevi.

EVA raggiunge o supera la media di training in 23 generazioni su 36. Nell’ultima
generazione è inferiore alla media in 3 run su 6: il grafico mantiene quei valori,
senza sostituirli con la media o con il migliore esploratore. Nel test finale è
superiore alla media della popolazione in 5 run su 6. Rispetto alla propria
valutazione iniziale migliora in 4 run e peggiora in 2: seed 801 da 42,50% a
26,25%, seed 812 da 40,00% a 38,75%.

Il campione è piccolo e i risultati sono correlati tra le due run della stessa
linea. Non prova miglioramento generale o monotono. La selezione degli insegnanti,
la capacità delle memorie, il credito terminale e l’interferenza tra contesti
rimangono limiti. In particolare, più apprendimento sul training non equivale a
maggiore capacità su ambienti nuovi. I due peggioramenti sono riportati
esplicitamente; non sono stati nascosti né corretti cambiando i test.

Tempo totale osservato circa 130 secondi nell’ambiente condiviso; non costituisce
un benchmark prestazionale comparabile con altre versioni o con il PC dell’utente.

## Pacchetto

Lo script di rilascio verifica CRC ZIP, checksum di ogni file, corrispondenza del
manifest e un’unica radice `predatore/`. Il manifest è nella stessa cartella dei
sorgenti. Nessuna cartella aggiuntiva `predatore_v6`, nessuna EvolutionVault
precompilata, nessun motore selezionabile delle altre versioni.
