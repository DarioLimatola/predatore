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
