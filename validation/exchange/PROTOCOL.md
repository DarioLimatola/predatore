# Prova di accettazione degli scambi

Prima delle misure si fissano due confronti distinti. Il ricevente è il circuito iniziale della run di riferimento V6.05 (seed 271, 32 moduli, mappa 8×8). Si confrontano un circuito della stessa run dopo due generazioni (seed di validazione 19001) e un controllo positivo composto a mano (seed 19002). Il controllo usa le direzioni osservabili del sensore e la normale inferenza LUT, con precedenza all'asse orizzontale. In assenza di informazioni si muove verso nord. Serve a verificare se il valutatore riconosce un candidato utile. Il suo comportamento è stato programmato.

Per ciascun confronto i quattro scenari classici hanno una mappa nuova e 32 condizioni di partenza casuali, con budget 64 azioni. Il controllore è congelato durante la misura. La soglia richiede un guadagno netto di catture aggregato, nessun calo netto in alcuno scenario e una coda esatta unilaterale sulle coppie discordanti ≤ 0,05. L'unità osservata è un episodio sulla rispettiva mappa fissata. Il confronto non misura tutte le geometrie o gli effetti dell'addestramento continuo. Il ricevente e il profilo persistente restano separati dai risultati di validazione.

**Esito del controllo di compatibilità prima delle misure.** Il donatore evoluto ha una maschera HDC 32766, il destinatario iniziale 32767. Il contratto vieta il passaggio grezzo. Per il confronto sul seed 19001 si trasferisce quindi un episodio di esperienza con sensori nominati, acquisito dal donatore in Moving su mappa 271, seed episodio 507, con budget 64. Il candidato grezzo rifiutato resta riproducibile.

| Trasferimento | Semi e budget | Catture del ricevente | Catture del candidato | Esito |
|---|---|---:|---:|---|
| Circuito V6.05 cresciuto nella run | Verifica del contratto all'importazione | — | — | Incompatibile per maschera HDC |
| Esperienza nominata dello stesso donatore | 19001, 4 × 32 episodi, 64 passi | 5/128 | 5/128 | Rifiutata, nessuna coppia discordante |
| Controllo positivo LUT basato sui sensori | 19002, 4 × 32 episodi, 64 passi | 10/128 | 63/128 | Accettato per questa prova |

Nel controllo positivo le catture guadagnate/perse sono 16/0 in Static, 16/1 in Moving, 10/0 in Maze e 13/1 in Hidden. L'esito positivo riguarda la regola direzionale importata. Per un modulo neurale appreso da Monade o Tesseract rimane aperta la verifica su mappe nuove. Il gate non installa i candidati nel profilo. I risultati JSON contengono i budget, gli esiti accoppiati, i tempi di esecuzione e le mappe.

## Riprodurre il controllo positivo

Dalla radice dello ZIP eseguire il comando seguente, scegliendo un percorso libero per il report.

```console
python -m tools.candidate_exchange evaluate validation/exchange/receiver_v605.json prova-lut.json --candidate validation/exchange/positive_control_candidate.json --seed 19002 --episodes 32 --steps 64
```

Per la seconda prova usare il candidato `experience_candidate.json` e il seed 19001. Per verificare l'incompatibilità usare `trained_candidate.json`. Per esportare un altro cervello o un episodio appena simulato consultare `python -m tools.candidate_exchange --help` e l'aiuto del rispettivo sottocomando.

## Formato e limiti

Il trasporto tensoriale usa `predatore.tensor/1`, ordine row-major, bit meno significativo per primo e parole da 64 bit little-endian. La forma e i nomi degli assi determinano la disposizione logica. L'ultima Word è completata con zeri. Valore e maschera dei bit conosciuti sono campi distinti. Il test nativo legge la stessa fixture da 72 bit usata da Python attraverso `monade_core::word::Word::from_bytes`. La dipendenza è il crate monade-core 0.6.5 alla revisione d6e1b8a99b408a288529d059c81551a9135a5ffc. Questa revisione offre Word, Size, Ref, Arena, Space e Tally per bit. Il descrittore tensoriale di trasporto e il contatore firmato per relazione sono specifici di Predatore. Il valore logico con stato sconosciuto resta esplicito anche quando un consumatore accetta soltanto lane note.

Il tipo `boolean-vector.circuit/1` scambia grafo, porte, operatori, tick, LUT e maschere di scrittura. Il destinatario valuta una copia con memorie nuove. Il codec rifiuta cervelli che dipendono da routing strutturale e indirizzi runtime. Per moduli con sensori, azioni o feedback di semantica differente occorre un nuovo adattatore esplicito. Un evento `arena.experience/1` usa nomi stabili, storia osservabile e tracce terminali dichiarate dal docente. Un singolo episodio può essere insufficiente per il consenso contestuale. Il campo di provenienza e la licenza sono dichiarazioni del produttore. SHA-256 controlla l'integrità e identifica la proposta. Il contenuto del report di validazione è generato dal destinatario.

I file del formato V6.05 sono leggibili come sorgenti e restano intatti. `receiver_v605.json` è la copia del cervello iniziale della run di riferimento. Un risultato accettato include la copia esportabile precedente alla validazione e l'identificatore del ricevente per tornare alla configurazione di partenza.
