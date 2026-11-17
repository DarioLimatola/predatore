# Rapporto di verifica V6.05

## Protocollo

Tre seed e tre budget fissi: 32, 64, 96 moduli da 64 bit. Per confronto gli altri parametri sono identici: 6 esploratori, 10 generazioni, 8 episodi di training per generazione, 8 episodi di validazione, 32 episodi di holdout, arene 2D 8×8 e 60 passi massimi per episodio.

L'holdout confronta EVA iniziale, EVA finale e popolazione finale con copie adattive su stessi episodi. Nessun feedback o seed di holdout entra nella selezione. I processi sono eseguiti in sequenza, con memoria misurata isolatamente nei nove casi del confronto.

## Confronto dei moduli

| Moduli | EVA iniziale | EVA finale | Pop. finale | Tempo | Picco RAM |
| --- | --- | --- | --- | --- | --- |
| 32 | 16,67% | 18,75% | 17,19% | 13.5 s | 51.6 MiB |
| 64 | 19,79% | 26,04% | 27,26% | 16.8 s | 54.7 MiB |
| 96 | 26,04% | 29,17% | 30,38% | 19.8 s | 57.3 MiB |

96 moduli hanno la media EVA finale più alta in questo campione. EVA regredisce in 4/9 casi e chiude sotto la media degli esploratori in 5/9. Le regressioni di validazione dopo il training personale sono 62/90 generazioni. Tre seed non bastano per scegliere una dimensione generale né per cambiare i default 32/64/128.

| Moduli | Seed | EVA iniziale | EVA finale | Pop. finale | Eseguiti EVA |
| --- | --- | --- | --- | --- | --- |
| 32 | 42 | 12,50% | 6,25% | 11,98% | 4 |
| 32 | 117 | 12,50% | 18,75% | 14,58% | 15 |
| 32 | 809 | 25,00% | 31,25% | 25,00% | 9 |
| 64 | 42 | 18,75% | 18,75% | 21,35% | 18 |
| 64 | 117 | 31,25% | 28,12% | 29,17% | 25 |
| 64 | 809 | 9,38% | 31,25% | 31,25% | 7 |
| 96 | 42 | 6,25% | 25,00% | 23,44% | 12 |
| 96 | 117 | 43,75% | 40,62% | 44,79% | 27 |
| 96 | 809 | 28,12% | 21,88% | 22,92% | 35 |

## Continuità tra run

| Linea | Seed | EVA iniziale | EVA finale | Differenza |
| --- | --- | --- | --- | --- |
| 801 | 801 | 37,50% | 43,75% | +6,25 pp |
| 801 | 802 | 50,00% | 46,88% | -3,12 pp |
| 811 | 811 | 21,88% | 18,75% | -3,12 pp |
| 811 | 812 | 28,12% | 25,00% | -3,12 pp |

L'identità dello snapshot finale della prima run coincide con quella iniziale della seconda in entrambe le linee. La prestazione iniziale/finale si confronta entro ogni run; seed diversi cambiano il test. La persistenza è stata verificata, la crescita monotona no.

## Prestazioni

| Moduli | Eseguiti | Completo | Sparso | Rapporto |
| --- | --- | --- | --- | --- |
| 32 | 9 | 113.5 ms | 39.1 ms | 2.90× |
| 64 | 18 | 216.9 ms | 66.4 ms | 3.27× |
| 96 | 5 | 350.1 ms | 30.0 ms | 11.67× |

Mediana di tre ripetizioni per 1500 decisioni identiche (seed del circuito 42, decoder di memoria disattivato, senza feedback). Le azioni coincidono tra esecuzione completa e sparsa. Il numero di moduli eseguiti dipende dai collegamenti e non cresce necessariamente con il totale.

Profilo cProfile separato: 8.58 s cumulativi in rollout, 4.80 s in decide e 5.21 s in atomic_json. Sono tempi inclusivi e non si sommano. La simulazione, la validazione e la serializzazione incidono sul tempo complessivo; accelerare una singola decisione non produce lo stesso guadagno su tutta la run.

Nei nove training la decodifica finale HDC compare 331 volte su 27709 decisioni di EVA. La frequenza misura l'uso, non il beneficio causale. Le ablation nelle run tengono fissi i seed ma non garantiscono generalizzazione.

## Test e riproduzione

104 test riusciti; 1 saltato (Tk nativo richiede un display). Verificati callback GUI, ripresa deterministica, persistenza tra run, memoria, distillazione e corrispondenza del calcolo sparso con quello completo. Esempio XOR: 50% prima del feedback e 100% sui quattro casi dopo il training.

    python -m unittest discover -s tests -v
    python tools/benchmark_release.py
    python examples/binary_task.py

Risultati individuali e configurazioni: controlled.json, continuity.json, inference.json, environment.json, profile.txt e profile_run.json. Le misure si riferiscono al sorgente ricostruito e verificato in questa consegna. Il benchmark include il costo di salvataggio, valutazioni e diagnostica.
