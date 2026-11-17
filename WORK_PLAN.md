# Rilascio 6.05 — contratto di lavoro

Obiettivo: consolidare l'applicazione Predatore e separare un engine booleano
vettoriale riusabile. Una sola implementazione e formato corrente, nessun motore
legacy selezionabile. Nessun Rust, parallelismo o nuovo dominio/scenario 3D o
linguistico. Le verifiche delle primitive dimensionali non sono nuovi scenari.

## Funzioni da consegnare

- Core autonomo `boolean_engine`: osservazioni binarie e azioni configurabili,
  feedback e serializzazione; nessuna dipendenza da mondo, GUI o filesystem.
- Esecuzione del sottografo raggiungibile con semantica sincrona invariata,
  ritardi, gate e cache invalidata alle mutazioni. Moduli dormienti conservati.
- RAM breve e contestuale con evidenze a 16 bit, canali personali/terminali/
  docenti, provenienza limitata, deduplicazione, invecchiamento e budget finiti.
- HDC/VSA: binding XOR, maggioranza, permutazioni, codifica delle caratteristiche
  e della sequenza, recupero Hamming con consenso e astensione.
- Operatori per asse, vicinati e ritardi; evoluzione di forma e rappresentazione
  entro budget. Più assi non significano automaticamente più capacità.
- Esploratori adattivi, EVA sempre adattiva e consolidamento dei migliori,
  SMITH con archivio e scelta contestuale fra strategie definite e misurate.
- Training deterministico variato per generazione più replay solo di training;
  validazione e test isolati, continuità e ripresa deterministica.
- GUI conservata, senza scheda Vault; moduli eseguiti/totali e uso delle memorie
  visibili; confronto dei circuiti selezionati.
- Confronto controllato 32/64/96, profilo costi e prove di apprendimento sulle
  arene 2D esistenti. Risultati negativi riportati senza modificare i test.
- Manuale completo in italiano: uso, algoritmi, strutture, API, formule,
  esempi, limiti, verifica, risorse e guida per riusare il core.

## Esclusioni esplicite e limiti scientifici

Non viene promesso miglioramento monotono, intelligenza generale, semantica
linguistica, nove livelli generali di metacognizione o equivalenza con un LLM.
La supervisione sceglie fra strategie programmate: non inventa algoritmi.
Le idee iniziali di plasticità hebbiana e rumore probabilistico non sono
requisiti sostitutivi della plasticità booleana e delle mutazioni già adottate.

## Avanzamento

- [x] Base V6.04 aggiornata, con rimozione del pannello Vault.
- [x] Core e primitive dimensionali/HDC.
- [x] Memorie e distillazione.
- [x] Training, EVA e SMITH.
- [x] GUI e regressioni.
- [x] Benchmark e profilo.
- [x] Documentazione e pacchetto verificato.
