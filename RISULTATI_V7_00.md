# Predatore V7.0 · Risultati

**Rilascio del 20 novembre 2026.** La V7 rende il training più rapido senza cambiarne i risultati. Le decisioni di partenza sono nella seduta 18 del consiglio.

## Che cosa cambia

| Area | V6.06 | V7.0 |
|---|---|---|
| Rust | Sonda LUT separata, un processo per episodio, mai usata nel training | Modulo `predatore_native` caricato da Python: episodi interi (mondo, preda, decisione, memorie, HDC, feedback, plasticità) |
| Memorie | Oggetti Python, copiate via JSON | Oggetti Rust con la stessa interfaccia; copie dirette; impronte con digest SHA-256 in cache |
| Parallelismo | Nessuno | 16 thread (configurabile); GIL rilasciato durante gli episodi; esploratori e valutazioni in parallelo |
| Generatore degli episodi | Mersenne Twister di Python | xoshiro256** / SplitMix64 da `tesseract-core`, identico in Python e Rust |
| Arena | Live, Simulazione agenti, Replay del test finale | Live e Registrazioni: training reale di tutti gli agenti a ogni generazione, più il test finale etichettato TEST |
| Brain Inspector | «Confronta con cervello JSON…» | EVA prima/dopo: rispetto alla generazione precedente o all’inizio della run |

## Equivalenza: le misure sono le stesse

- **Episodio per episodio** (`tests/test_v7_native_parity.py`): stessi sette scenari, la configurazione predefinita più sei varianti del cervello. Variano larghezza 8–256 bit, operatori tensoriali e ritardi, 3 tick, codifica broadcast, memoria spenta e storia di 6 passi. In tutti i 238 casi Python e Rust danno le stesse metriche, gli stessi eventi di esperienza e lo stesso export completo, stato di lavoro compreso.
- **Run intere** (`tools/verify_v7_equivalence.py`, 3 run consecutive da 5 generazioni). Python puro, nativo con 1 thread e nativo con 2 thread producono file identici: profilo SMITH completo, benchmark e fotogrammi di tutte le 195 registrazioni.
- **Suite completa:** 147 test superati, GUI inclusa (Tk reale sotto Xvfb). Senza modulo nativo i test superati sono 141, con 6 saltati: sono i test che richiedono il kernel.

## Velocità

Le misure vengono da una macchina Linux con **2 thread** e Python 3.13, con le impostazioni predefinite della GUI: popolazione 12, 100 passi, 24 episodi di validazione, 100 di test. Sul tuo PC i numeri saranno diversi: **MISURA_V7.bat** li ricava lì.

| Misura | V6.06 | V7 Python | V7 nativo, 1 thread | V7 nativo, 2 thread |
|---|---|---|---|---|
| Una run da 30 generazioni, compresa la valutazione finale | 243 s | 218 s | 53,7 s | 48,9 s |
| Costo per passo di episodio | ~150 µs | ~150 µs | 5–8 µs | — |

| Run consecutive sullo stesso profilo (3 generazioni l’una) | 1ª | 3ª | 6ª | 9ª |
|---|---|---|---|---|
| V6.06 | 19,6 s | 37,4 s | 60,3 s | 69,0 s |
| V7 nativo, 2 thread | 4,4 s | 7,2 s | 8,5 s | 8,9 s |

Nella V6.06 il tempo cresceva con il numero di run perché ogni generazione rivaluta i compiti di validazione delle run precedenti (fino a 8). Nella V7 quella crescita è contenuta.

**Dove va ora il tempo:** l’episodio nativo pesa circa il 15% di una run. Il resto è orchestrazione Python:
- salvataggi JSON dei checkpoint;
- distillazione;
- valutazione finale;
- diagnosi.

Su 2 thread il parallelismo aggiunge poco, perché quella parte resta seriale. Con 20 thread il guadagno dipende dalla quota parallela e va misurato sul posto.

## Registrazioni e GUI

- A ogni generazione si registra un episodio di training reale per EVA e per ciascun esploratore. Lo scenario ruota fra le generazioni ed è lo stesso episodio per tutti gli agenti. L’episodio registrato passa dal percorso Python, che è identico al nativo, quindi i risultati non cambiano.
- Si conservano le ultime 5 generazioni e la prima: circa 50–250 kB per episodio.
- La vista Live mostra ogni generazione un esploratore (a rotazione, o quello scelto) ed EVA. Gli esploratori si addestrano in parallelo, quindi non possono essere osservati tutti dal vivo nello stesso momento. Tutti sono nelle Registrazioni.

## Compatibilità

- I checkpoint e i profili della V6.06 restano leggibili: il formato del cervello non cambia.
- Una run V6 ripresa con la V7 prosegue con il nuovo generatore. È una linea V7 e si confronta con la V6 solo sulle statistiche.
- Gli ID dei genomi cambiano: struttura più digest delle memorie.
- La sonda Rust della V6.06 resta in `legacy/v6_lut_probe/` con il suo protocollo e i dati golden.

## Non verificato in questo rilascio

- **Compilazione e misure su Windows.** Qui non si raggiungono le librerie Rust per Windows. Il modulo si compila con COMPILA_NATIVO.bat oppure con il workflow GitHub `native-wheels`, che esegue anche i test di parità.
- **Scala oltre 2 thread**, e il comportamento con i core ibridi (prestazionali ed efficienti) di Windows.
- **L’esperimento proposto dalla voce di minoranza** (più seed, curve lunghe, diagnosi delle regressioni di EVA): è la tappa V7-F, ora realizzabile in tempi ragionevoli.
