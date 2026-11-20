# P2 · Risultati e decisione del 20 novembre 2026

**Decisione**. Il laboratorio continua a usare Python per il training. Il prototipo Rust è compilabile e riproduce esattamente il circuito coperto dal corpus. Il vantaggio sul tempo end-to-end è inferiore alla soglia del 10%. La variante resta disponibile come esperimento autonomo.

Il profilo di un run V6.05 su seed 42, 64 moduli e una generazione ha registrato 20.680 chiamate a `decide`, per 3,83 s cumulativi su 11,35 s sotto profiler. Il tempo cumulativo include decoder e memoria oltre alle LUT. Checkpoint e salvataggi costituivano altri costi rilevanti: il raddoppio della velocità del solo circuito non si tradurrebbe nel raddoppio del throughput di training.

**Parità osservata**. Dodici episodi V6.05, tre seed di rete, quattro scenari storici e 289 decisioni con tutte le parole di stato e le azioni identiche. Per i nove episodi con preda ferma anche le traiettorie del mondo, i reward, le collisioni e le terminazioni concordano a ciclo chiuso. Per i tre episodi Moving si è controllata l'inferenza sulla sequenza sensoriale originale. Il corpus è diagnostico e impiega reti congelate.

| Misura | Python | Rust con driver Python |
|---|---:|---:|
| Tempo mediano per gruppo di nove episodi, 31 repliche | 21,180 ms | 21,070 ms |
| CPU totale in un processo fresco, 31 gruppi | 0,670 s | 0,842 s |
| Picco RSS del driver | 15,125 MiB | 15,125 MiB |
| Massimo RSS osservato di un figlio | — | 14,750 MiB |

I tempi mediani delle singole fasi native sono 10,277 ms per tabella e serializzazione, 10,018 ms per avvio/IPC/calcolo e 0,761 ms per il punteggio nel mondo. La mediana della somma si calcola sui gruppi completi. Il rapporto Python/Rust è 1,005×. In due misure preliminari era 0,847× e 1,034×. La macchina Linux era condivisa con altre prove e l'evidenza non giustifica una promozione; il training completo con decodifica associativa, apprendimenti online e salvataggi rimane Python. Non è stata misurata una versione Rust di quel training completo.

**Riproducibilità**. `PROTOCOLLO.md` descrive comandi, golden, limiti e rollback. `benchmark.json` conserva i tempi di ogni episodio e replica. Verificati con Rust 1.97.0, Cargo con lock, Python 3.12.14 su Linux x86_64. Il build release è riuscito e i due test Cargo sono passati, incluso il confronto delle parole con la maschera di conoscenza Monade. I test Python `test_native_probe.py` sono passati sia per il riferimento sia per l'eseguibile reale.
