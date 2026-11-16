# V6.04

- EVA - learning agent svolge training reale a ogni generazione e conserva gli
  aggiornamenti anche quando il trasferimento di un insegnante viene respinto.
- SMITH - agent supervisor possiede archivio, specialisti e metadati delle
  strategie. Profilo, cartella e schema persistente corrispondono a questo ruolo.
- Distillazione dai migliori esploratori correnti e da precedenti insegnanti;
  le esperienze degli altri agenti rimangono campionabili nell’archivio SMITH.
- Il ranking di trasferimento bilancia compiti correnti e ricordati. Rimosso il
  veto di qualsiasi singola regressione di scenario; log espliciti delle perdite.
- Confronto di training omogeneo: EVA e popolazione apprendono durante gli stessi
  episodi. Validazione/test usano copie adattive isolate, senza agenti duplicati.
- Base/min/max predefiniti 64/32/128, tre slider con toggle e campo intero;
  variazioni strutturali ±1 nei discendenti. Eliminati i timer di crescita/potatura.
- Blu EVA, verde media popolazione, arancio validazione; legenda generata dalle
  serie tracciate e diversa per ogni metrica. Dati per scenario completi.
- Pannelli EVA/SMITH legati al checkpoint della run selezionata, non all’ultimo
  profilo globale. Preview degli esploratori realmente valutati in quella generazione.
- Live distingue lo slot EVA; un solo export `brains/eva.json` durante/dopo la run.
- Corretto il manifest ZIP che creava la cartella aggiuntiva `predatore_v6`.
  Un’unica cartella `predatore/`; formati correnti senza importatori delle altre versioni.
- Test di ripresa, isolamento, osservabilità delle catture, trasferimento,
  rimozione/rimappatura dei neuroni, slider, cambio run/profilo e packaging.

Il miglioramento delle metriche non è garantito. Le prove di comportamento e i
risultati, compresi eventuali peggioramenti su mappe nuove, sono riportati in
`validation/v604/REPORT.md`.
