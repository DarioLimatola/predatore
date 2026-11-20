# V7.0 · Nucleo Rust nel training, parallelismo, arena reale

- `native/` (crate `predatore-native`, modulo Python `predatore_native`, PyO3): episodi interi in Rust. Copre mondo e preda, decisione, memoria contestuale e breve, memoria associativa HDC, previsioni, evidenze di circuito, feedback, plasticità ed esperienze. Il riuso è diretto: `tesseract-core` per il generatore, `monade-core` per le fondazioni, entrambi a revisione fissata.
- Memorie native persistenti con la stessa interfaccia Python. Le copie dei cervelli non passano più dal JSON (`clone`). L’impronta del cervello usa i digest SHA-256 delle memorie, calcolati e messi in cache in Rust.
- Generatore degli episodi xoshiro256** (SplitMix64), identico in Python e in Rust. È una nuova linea sperimentale dichiarata e gli ID dei genomi cambiano rispetto alla V6.
- Parallelismo: 16 thread di default (`PREDATORE_WORKERS`), rilascio del GIL durante gli episodi, esploratori e valutazioni in parallelo. I risultati sono identici con qualunque numero di thread. Un eventuale ripiego del pool viene segnalato.
- Registrazioni: un episodio di training reale per EVA e per ogni esploratore a ogni generazione (`recordings/`), conservate per le ultime 5 generazioni e per la prima.
- GUI: modi Live e Registrazioni. Rimossa la Simulazione agenti. Aggiunti la vista affiancata degli agenti di una generazione e il pannello «Decisione». Il test finale è etichettato TEST. Nel Brain Inspector il confronto EVA prima/dopo sostituisce «Confronta con cervello JSON».
- Strumenti: `tools/benchmark_v7.py` (MISURA_V7.bat), `tools/verify_v7_equivalence.py`, COMPILA_NATIVO.bat, workflow `native-wheels`, `launcher.py --engine-status`.
- La sonda LUT della V6.06 si sposta in `legacy/v6_lut_probe/`. Il suo test golden V6.05 dipende dal generatore V6 e non fa parte della suite V7.

# V6.06 · Laboratorio comportamentale

Curriculum esteso, preda neurale, credito temporale opzionale, protezione di EVA per scenario, verifica dei candidati, migrazione dei checkpoint e nuova navigazione dell’arena. Per opzioni, risultati e compatibilità vedere README.md e RISULTATI_V6_06.md.

# Predatore V6.05

- Isolato il motore riusabile in boolean_engine. Il simulatore conserva un
  solo backend, con osservazioni binarie e numero configurabile di bit di azione.
- Esecuzione sparsa sincrona, memoria dei ritardi e operatori per assi.
  Mutazioni di rappresentazione e limiti congiunti di moduli e larghezza.
- RAM a 16 bit con canali distinti, provenienza, decadimento, trasferimenti
  deduplicati e fallback HDC con consenso e astensione.
- Nuovo training deterministico per generazione; replay soltanto di training.
  SMITH sceglie fra quattro strategie e raccoglie il risultato a fine generazione.
- EVA resta adattiva e confronta proposte dei docenti su copie isolate.
  La regressione dopo l'apprendimento personale è misurata separatamente.
- GUI senza scheda Vault; confronto dei circuiti, origine del decoder,
  conteggio moduli eseguiti e memorie. Impostazione della larghezza massima.
- Report 2D riproducibile, profilo costi, manuale in italiano ed esempio API.

Formati nuovi, senza migrazione dei profili precedenti. Nessun nuovo scenario
multidimensionale, nessun Rust e nessun calcolo parallelo.
