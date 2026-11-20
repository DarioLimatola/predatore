# PREDATORE — Piano di modifiche per il laboratorio evolutivo e comportamentale

**Documento operativo autonomo per la finestra di sviluppo di Predatore**  
**Data di riferimento:** 19 novembre 2026 · **Stato:** specifica tecnica; non attesta nuove implementazioni.  
**Repository:** `DarioLimatola/predatore` — V6.05 al momento dell'audit, ramo corrente da verificare all'apertura.  
**Fondazione da riusare:** `monade-core` (Rust) per struttura binaria, vettori, memoria e riferimenti tramite API pubblica.  
**Scopo esclusivo:** apprendimento evolutivo e continuo degli agenti, decisioni nel mondo, plasticità, mutazioni, credito temporale, trasferimento delle competenze e simulazioni visive. **Fuori scope:** elaborazione del linguaggio, chatbot, crawler letterario, sistema cognitivo generale.

> **Mandato per l'assistente di sviluppo.** Lavora sui sorgenti reali dell'ultima revisione di Predatore, non partire dal vecchio ZIP V5.03 come base di nuova versione se V6.05 è presente. Prima verifica il codice, i test, gli esperimenti e i checkpoint. Cambia solo i componenti necessari, senza rimuovere GUI, EVA, SMITH, EvolutionVault, scenari, replay, benchmark, salvataggi e telemetria già funzionanti. Ogni porting deve essere confrontato con la V6.05 sugli stessi episodi/seeds e avere una via di ripristino. Non riscrivere tutto il software in un'unica modifica.

## 1. Identità del progetto e ruoli

**Predatore è il laboratorio che studia come una popolazione di agenti migliori attraverso interazione con l'ambiente.** L'efficacia si misura nel comportamento e nel trasferimento di abilità, non soltanto nella complessità del circuito.

- **EVA:** agente principale persistente; impara da proprie esperienze e può adottare contributi verificati di altri agenti. Non va confusa con copie effimere di validazione. Non garantire matematicamente che EVA sia sempre migliore della media: misurare come e quando regredisce.
- **SMITH:** supervisore con memoria tra run e selezione di procedure/curricula; non è attore dell'arena né un generatore autonomo di algoritmi che non esistono.
- **Esploratori:** popolazione di agenti che sperimenta strategie, topologie, circuiti e rappresentazioni; propongono insegnamenti a EVA sotto validazione.
- **Ambiente:** scenari controllabili, riproducibili, con preda, ostacoli, visibilità e feedback. Le azioni visibili devono derivare dalla rete effettivamente in valutazione, non da regole della GUI.

Monade fornisce i primitivi strutturali; Tesseract fornisce risultati sperimentali sui moduli neurali e sul credito profondo. Predatore resta il luogo dove misurare se tutto ciò migliora la capacità di agire, adattarsi e imparare dall'esperienza.

## 2. Baseline: che cosa c'è già nella V6.05

| Componente | Sorgente e comportamento riscontrato | Decisione |
|---|---|---|
| `boolean_engine/network.py` | Grafo vettoriale binario, LUT per lane, ricorrenza, memoria, plasticità, calcolo sparso degli antenati | **Preservare come riferimento; portare in Rust per fasi misurate** |
| `boolean_engine/tensor.py` | Rotazioni, reshape e trasformazioni di forme variabili a numero di bit costante | **Mantenere; adattare alla struttura tensoriale comune** |
| `boolean_engine/context.py` | Chiavi da osservazione e storia | **Mantenere; evolvere verso riferimenti/temporalità esplicita** |
| `boolean_engine/memory.py` | Evidenze contestuali, canali e decadimento, fonti | **Mantenere, confrontare con contatori strutturali** |
| `boolean_engine/hdc.py` | Binding XOR, bundle, rotazione, ricerca per Hamming e memoria associativa | **Riutilizzare, distinguendo bucket di contenuto e indirizzi strutturali** |
| `boolean_engine/distillation.py` | Contributi con provenance, deduplicazione e voto fra fonti | **Preservare come meccanismo di insegnamento** |
| `engine/continual.py` | Training evolutivo, trasferimenti, EVA persistente, validazione/holdout isolati | **Preservare e correggere regressioni con gate più robusti** |
| `engine/smith.py` | Meta-selezione di strategie implementate e continuità delle run | **Mantenere; studiare meta-apprendimento verificabile** |
| `engine/world.py`, `challenges.py` | Arene 2D e scenari riproducibili | **Mantenere e migliorare varietà/difficoltà dei mondi** |
| `engine/telemetry.py` | `LiveObserver`, telemetry reale con perdita controllata di frame | **Non ricreare: migliorare il prodotto e il costo** |
| `gui/laboratory.py`, `brain_inspector.py` | Live Training, Independent Preview, Replay e Brain Inspector | **Riprogettare l'interazione e la leggibilità; non fingere che manchino** |
| `engine/vault.py` | Checkpoint, snapshot, salvataggi atomici e verifica | **Preservare con migrazioni versionate** |

**Dati documentati, non rieseguiti in quest'audit:** 104 test riusciti e uno saltato per display nativo. Nove piccole prove controllate (3 seed × 32/64/96 moduli): EVA regressa 4/9; chiude sotto la media degli esploratori 5/9; miglioramenti non monotoni in continuità. I benchmark di inferenza sparsa sono favorevoli, ma il profilo cita anche simulazione, validazione e salvataggi come costi importanti. Quindi dire «Rust = tutto immediatamente più veloce» non è verificato.

**V5.03 storica:** archivio importante per confronti e non regressione; non reintrodurre la dualità legacy/recovery o la vecchia simulazione di preview al posto del Live Training reale.

## 3. Principi per le modifiche

1. **Evoluzione reale, non decorativa:** i discendenti ereditano, esplorano e vengono valutati su episodi fissati; non chiamare apprendimento la semplice registrazione delle mutazioni.
2. **Mutazione ≠ plasticità:** mutazione modifica bit/maschere/strutture del circuito o la codifica; plasticità modifica collegamenti, routing, intensità di relazione e criteri di attivazione fra moduli/tensori. Entrambe possono avvenire, ma i loro effetti vanno separati nel log.
3. **Apprendimento continuo senza leakage:** gli episodi di training possono modificare la rete; le copie usate in validation/holdout non restituiscono esperienze al profilo persistente. Conservare provenance e condizioni sperimentali.
4. **Contatori strutturali da 16 a 64 bit:** candidati sulle strutture/relazioni attive, larghezza variabile su necessità e non su dimensioni/tensor rank. Gli attuali `i16`/evidenze contestuali e la variante per-lane restano controlli; non adottare un solo contatore globale quando il compito richiede molte decisioni indipendenti.
5. **Riuso diretto del core condiviso:** nessuna seconda implementazione concorrente di Word/Address/Space/Hash, nessun accesso ai dettagli privati della mente.
6. **Tutti i compiti sono di comportamento, geometria, memoria e pianificazione:** niente lingua o formazione generalista nell'agente predatore.
7. **Interfaccia pulita e scientifica:** mostra ciò che accade davvero e consente di capire *perché* è accaduto.

## 4. Struttura software futura: motore Rust + laboratorio separato

**Decisione di direzione:** sì al Rust per il percorso computazionale intensivo, **ma dopo profiling e con migrazione incrementale**.

### 4.1 Ruoli dei livelli software

- **`monade-core` (dipendenza Rust, pin-nata):** word/bitvec, algebra, indirizzi, descrittori tensoriali e memoria strutturale tramite API stabile, nessun accesso mutabile a fondazioni o seed.
- **Backend Rust Predatore (nuovo, delimitato):** inferenza, calcolo vettoriale/LUT, tracing selettivo, update dei collegamenti, eventuale ciclo episodico, valutazione, RNG riproducibile e simulazione se è effettivamente un collo di bottiglia.
- **Orchestrazione esperimenti:** generazioni, configurazione, ranking, scelte di SMITH, curricoli, commit dei checkpoint. Migrare in Rust quanto è misurato costoso; non per forza tutto nella prima fase.
- **Strato visuale:** GUI indipendente dai dettagli degli attori; riceve eventi reali del backend, snapshot e replay; può restare in Python inizialmente, poi essere sostituita in modo controllato.

**Confinare la frontiera fra Rust e Python:** evitare una chiamata Python↔Rust per singolo bit, singolo gate o singolo tick. Preferire l'esecuzione di episodi interi in Rust con eventi/aggregati serializzati e comandi di controllo a grana grossa. Considerare binding o subprocess solo dopo benchmark dell'overhead. Mantieni il vecchio backend Python in una suite comparativa/di migrazione finché la parità è verificata, senza presentarlo come toggle legacy permanente.

### 4.2 Regola di porting

Per ogni componente: snapshot golden degli input, degli stati, delle azioni, delle ricompense e degli update; porting; confronto su stessi seed; profiling del costo **totale** per run. Se la scelta del generatore casuale cambia, usare stream di episodi preregistrati come base comparabile e dichiarare il cambio; non pretendere un bitwise parity inesistente. Salvare reference e release per rollback.

**Ordine consigliato:** bit operations e tensor views → circuit inference/decoder → trace e plasticità → world step e rollout → generazioni/ranking/checkpoint soltanto se misurato necessario. Misurare CPU wall, RAM, save/load e throughput per stage.

## 5. Come incorporare la struttura di Monade

- **Rappresentazione:** moduli, stati e tensori possono essere supportati da Word/Size/descriptor comuni; una Monade da 2 MiB non è un neurone standard né la dimensione minima del cervello.
- **Riferimenti e collegamenti:** runtime refs per moduli attivi; indirizzi ottali per navigazione; ID persistenti per genome/snapshot; contenuti appresi non identificati dal solo seed.
- **Attivazione e plasticità:** collegamenti fra strutture referenziate, con contatore firmato opzionale per relazione, e maschere di attivazione/versione. Cambiare legami e routing è plasticità, cambiare LUT e bit di rappresentazione è mutazione.
- **Memoria:** usare query per riferimento, per contesto e per Hamming senza confondere l'indice di una scena con le chiavi associative. HDC/VSA esistenti restano baseline fino a confronto positivo.
- **Tempo:** rappresentare sequenze/periodicità/intervalli con vettori e relazioni temporali; mantenere anche tick esterni per il simulatore e la riproduzione. Un singolo clock non sostituisce il ragionamento sul tempo.
- **Provenienza e split:** trasferimenti importati soltanto come candidati con fonte, generazione, dominio, encoder, tipo di apprendimento, risultati e versioni.

**Compatibility gate:** scambio dei bit grezzi soltanto quando seed, hash, endian, packing, semantica degli assi, codifica dei sensori e limiti sono identici. Altrimenti trasferire esperienze strutturate/parametri descrittivi da ri-encodare. Un riferimento runtime del core non è serializzabile come ID persistente senza mapping.

## 6. EVA: imparare davvero senza regressioni nascoste

Separare quattro forme di apprendimento EVA: esperienza diretta, insegnamento contestuale/distillazione, adozione di un modello o modulo, e modifiche strutturali/di plasticità. Ogni canale deve avere contatori di prove, mutazioni validate, decisioni accettate/rifiutate, effetto per scenario e costo temporale.

**Regole:**

1. Ogni proposta viene valutata su episodi di validation non usati per crearla. L'adozione non usa holdout finale né esempi di test sigillato.
2. Tenere **ultimo checkpoint stabile** e candidato; attivare rollback se la regressione supera una soglia preregistrata. Non imporre un miglioramento monotono fittizio, misurare trade-off fra scenari.
3. Confrontare score aggregato, **per scenario**, success rate, tempo di cattura, fallimenti, traiettoria, energia/costo decisionale e difficoltà ambientale.
4. Diagnosi specifica delle 4/9 regressioni documentate e delle 62/90 regressioni di validazione post-training personale: verificare credito erroneo, plasticità troppo aggressiva, decoder che prevale sul circuito, scelta di teacher e varianza da seed.
5. Distillazione: la stessa esperienza ripetuta dallo stesso teacher non costituisce testimoni indipendenti; deduplicare per provenienza. Valutare anche fonti contraddittorie e casi rari.
6. Memoria fra run: verificare integrità, ricomparsa dei task precedenti, e trasferimento su nuove mappe. La semplice identità del checkpoint non dimostra miglioramento cognitivo.

**Questioni diagnostiche aperte:** EVA inferiore alla popolazione, esiti dello SMITH, memoria che prevale sul circuito, differenze di prestazione fra compiti. Registrare condizioni che producono il fenomeno e controesempi.

## 7. SMITH ed evoluzione

- Conservare lo storico del supervisore, ma distinguere scelte fra procedure esistenti da vera invenzione di strategie nuove.
- Fare evolvere **topologia, moduli attivabili, selezione dei sensori, dimensioni latenti, plasticità dei collegamenti, criteri di consolidamento, durata delle tracce e policy di esplorazione**, non soltanto flip casuali di LUT.
- Selezionare per **capacità di imparare** (curva e generalizzazione a compiti nuovi), non per punteggio statico in un episodio ripetuto.
- Mantenere novelty search, archivi di comportamenti, lineage genomi e preservazione di nicchie per evitare collasso della popolazione.
- Registrare separatamente miglioramenti di un individuo, media della popolazione, miglior agente, EVA e generalizzazione; la media può peggiorare durante l'esplorazione e non è da sola una prova di bug.
- Introdurre ablation per evoluzione, plasticità online, memoria contestuale, distillazione e topologia ricorrente per attribuire i benefici reali.

## 8. Deep learning e credito temporale nel comportamento

**Ipotesi da verificare, non già dimostrata:** moduli con contatori strutturali 16–64 bit, mutazioni locali e collegamenti plastici possono apprendere sequenze di decisioni più complesse di circuiti LUT privi di credit assignment profondo.

- **Baseline neurale:** circuito booleano V6.05 invariato, applicazione dei suoi feedback, soli cambiamenti topologici, decoder di memoria e moduli apprendimento alternativo importati come *candidati sperimentali*.
- **Prove:** azioni con conseguenze ritardate, preda nascosta che riappare, necessità di ricordare posizione/velocità, compiti che richiedono due o quattro trasformazioni causali interne; controllo con tracce disattivate.
- **BEP** (ICLR 2026) serve come baseline teorica per credito profondo supervisionato, non come garanzia di apprendere da soli reward globali; **e-prop/tracce** sono metodi da confrontare per il credito temporale. Rif. https://proceedings.iclr.cc/paper_files/paper/2026/hash/2aab54135bd206ef6d4949ce17528d98-Abstract-Conference.html e https://www.nature.com/articles/s41467-020-17236-y.
- La fonte del reward deve essere parte dell'ambiente e osservabile; non introdurre comportamenti 'corretti' artificiali nella GUI. I sensori devono rispettare il contratto dell'agente; i benchmark con informazione privilegiata vanno etichettati come **upper bound**, non paragonati ingenuamente.
- Mantenere la differenza fra 'ha agito', 'ha contribuito' e 'ha causato il successo': la traccia di attivazione è una *ipotesi di credito* e va verificata con perturbazioni/counterfactual controllati quando fattibile.

## 9. Suite di ambienti e prove visive

Preservare i quattro scenari di base esistenti per confrontabilità, ma renderli **distintivi e non banalmente memorabili**:

| Scenario | Capacità da verificare | Diagnosi visiva |
|---|---|---|
| Static | Ricerca e avvicinamento | Percorso, ostacoli e costo della cattura |
| Moving | Inseguimento/anticipazione | Previsioni vs movimento della preda |
| Maze | Vera navigazione con vicoli e deviazioni | Traiettoria, percorso possibile, blocchi |
| Hidden | Memoria della preda non osservabile | Ultimo avvistamento, decisioni senza visibilità |
| Intelligent prey (nuovo, opzionale) | Adattamento a una preda con politica apprendibile | Due reti distinte, evoluzione di ciascuna, co-adaptation |
| Delayed reward (nuovo) | Credito temporale | Segnali interni e tempo del feedback |
| nD structured tasks (facoltativi) | Ricombinazione di assi e trasformazioni, senza linguaggio | Proiezioni/slice della struttura e azioni |

**Test forti aggiuntivi:** mappe e seed mai visti; densità e labirinti variabili; prede con strategie avversarie; competenze apprese su 8×8 trasferite a mappe più grandi; nuovi ostacoli e visibilità; task nei quali memorizing fixed positions fallisce. Non mischiare scenari di difficulty diversa in una percentuale unica senza suddivisione.

**Statistiche per ogni scenario:** capture %, intervalli statistici, completion time medio/mediano, passi, ostacoli, collisioni, distanza dall'obiettivo, comportamento, uso della memoria, decisioni dominate da override contestuali e tempo per decisione. Baseline Random, Greedy **con stessi sensori** quando possibile; eventualmente upper-bound privileged separato.

## 10. Ridisegnare la GUI — eleganza, chiarezza, verità

La GUI attuale è giudicata troppo caotica. **Non cancellarla prima che la nuova interfaccia sia funzionalmente equivalente**. Riorganizzare i contenuti in una superficie coerente, con pochi pannelli, gerarchia tipografica chiara, colori informativi e nome delle metriche comprensibile.

**Obiettivo visivo:** un'arena leggibile e un diagramma della rete, con le informazioni essenziali sempre visibili: agente scelto, generazione, scenario, fase, dati veri vs preview vs replay, catture, tempi, RAM, versione del cervello.

**Flussi separati:**

- **Live Training:** frame reali ricevuti da `LiveObserver`, con id agente, generazione, scenario, azione, visibilità e motivazione del comportamento dove tracciabile.
- **Recorded Replay:** dati registrati, navigabili nel tempo e confrontabili fra agenti/strutture.
- **Independent Preview:** simulazione separata su genomi salvati, *esplicitamente etichettata* e senza implicare che sia il training live.

**Pannelli suggeriti:** (a) Esperimento/Training; (b) Arena e Replay; (c) Evoluzione/Benchmark; (d) Cervello e memoria; (e) Archivio/Confronto. Non necessariamente cinque tab obbligatorie: ridurre ridondanze e migliorare le decisioni operative.

**Brain Inspector:** far vedere moduli e collegamenti effettivi, valori/known masks, dimensioni, contatori strutturali attivi, LUT, segnali di credito, cambiamenti prima/dopo, blocchi inattivi/protetti. Filtri per livelli, colore per operazione/attività, zoom, selezione nodo, legenda. Non confondere grafici di popolazione con quelli di EVA.

**Vincolo di prestazione:** la visualizzazione non deve rallentare sensibilmente l'addestramento; telemetria bounded, sampled e mai sorgente di verità per fitness. Testare Windows, path spariti, file locked, stop/resume e GUI headless. Controlli con nomi chiari e numeri interi dove sono quantità discrete.

## 11. Conservazione di EvolutionVault e delle versioni

Non eliminarlo perché non centrale all'apprendimento. Definirne il compito chiaramente: **registro verificabile di esperimenti, configurazioni, checkpoint, baseline, metriche e lineage**.

- Snapshot e formato V6.05 sono riferimenti immutabili. I nuovi formati devono avere `schema_version`, `engine_version`, mappa, seed, sensori, azioni, architettura, origine dei candidati, revisioni, codec, endian e hash.
- Prevedere una **migrazione esplicita** dai checkpoint V6.05 alle nuove rappresentazioni ove fattibile. Non sovrascrivere vecchi profili; generare una nuova linea sperimentale con puntatore all'origine.
- Ripristino crash-resume su Windows: salvataggio atomico, verifiche hash, fallback a checkpoint precedentemente valido, riconciliazione log.
- Report narrativo e numerico, non solo elenco di file: cosa è stato sperimentato, quali competenze sono state apprese e su quali scenari si è registrato regresso.

## 12. Scambio di conoscenza e modelli

**Da Monade:** APIs di vettori/tensori/riferimenti/memoria e contatore 16–64 bit, senza copiarne lo stato privato; formato di proposta e provenance.

**Da Tesseract:** candidati di unità ternarie, attenzione, sensibilità booleana, apprendimento multilivello, nuovi test e tecniche di specializzazione. Importare un modulo **soltanto** se gli ingressi/uscite sono descritti, e la conversione preserva il significato degli assi; valutare separatamente prima di incorporarlo in EVA.

**Verso Tesseract:** casi riproducibili di credito differito, circuiti LUT, strategie evolutive, log di fallimento/apprendimento e confronti di plasticità. Il laboratorio neurale non deve diventare una copia dell'arena di caccia.

**Verso Monade:** eventi e relazioni validate, risultati su memoria temporale, metaparametri e procedure generalizzabili; non esportare una 'mente EVA' grezza assumendo che possa essere innestata in uno spazio generativo differente.

**Contratto di trasferimento:** `schema_version`, ID del tipo di evento/modulo, sensor/action contracts, bit length/packing, tensor axes, fingerprint seed/hash/codec, split, seed dell'episodio, versione del modello, provenienza, metadata di validazione, prove per scenario, costo e licenze. Flag `candidate`/`accepted`/`rejected`, rollback id. Non trasformare in aggiornamenti permanenti contributi ancora non verificati.

## 13. Piano incrementale con stop gate

**P0 — Fotografia della V6.05:** registrare revisione, test Python, esempi e benchmark di rilascio, tempi per stage, memorie e checkpoint. Eseguire smoke tests dei tre modi dell'Arena e di EVA/SMITH. **Gate:** inventario riproducibile, nessuna funzionalità persa.

**P1 — Adapter verso core comune:** descrittori tensor, bit packing, known masks, contatori strutturali opzionali, ID persistenti e serializer versionato; keep-old codec test. **Gate:** round-trip e incompatibilità esplicite, risultati V6.05 invariati dove previsto.

**P2 — Rust nel vero hot path:** portare inferenza LUT/sparse e poi eventualmente rollout completo con medesimo input stream; benchmark CPU, RAM, serializzazione, decisioni e tempo totale. **Gate:** vantaggio misurato o rollback. Migrazione Python→Rust non è di per sé un risultato.

**P3 — Nuovo apprendimento:** contatore 16–64 bit per struttura/relazione, plasticità dei collegamenti, mutazioni di bit, test controllati di credito e conflitto. **Gate:** miglioramento su compiti più difficili dei toy e ablation convincente.

**P4 — Evoluzione e protezione EVA:** cambiare mutazione/selezione/curriculum per premiare capacità di apprendere; anti-regressione, backup, distillazione robusta, comportamento di SMITH. **Gate:** retention e trasferimento, confronto per scenario, nessuna auto-valutazione su holdout usato per imparare.

**P5 — Mondi/compiti:** labirinti autentici, visibilità, preda intelligente separata come nuovo scenario, ritardi, mappe nuove. **Gate:** baseline strong, vera generalizzazione e dati osservabili.

**P6 — UI:** ristrutturazione elegante su eventi reali e backend esplicito, brain graph con contatori, replay e confronto. **Gate:** feature parity, nessuna confusione fra live/preview/replay, overhead misurato.

**P7 — Scambio esperienze:** formati, provenance, schema e import/export di modelli candidati con gate. **Gate:** nuovo modulo importato migliora realmente su episodi inediti; fallimenti documentati.

Ogni fase produce una release o PR autonomamente verificabile. Se una modifica fallisce il gate, conservarne l'esperimento ma non renderla automaticamente la nuova versione di riferimento.

## 14. Definition of Done

- Il progetto si avvia, allena, riprende da checkpoint e visualizza Live/Replay/Preview anche dopo le modifiche.
- EVA e SMITH mantengono stati persistenti e specificità di ruolo; nessun dato di validation/holdout torna nel training.
- Le percentuali sono sempre accompagnate da seed, scenario, budget, baseline pertinente, variabilità, tempo e costi.
- Il porting Rust garantisce parità funzionale dove possibile, o differenze dichiarate, ed è più veloce **nell'intero workload** misurato o motivato per altri benefici; non soltanto microbenchmark isolato.
- Il core comune è una dipendenza incapsulata, non duplicato; i genomi esistenti restano leggibili o migrabili con protocollo esplicito.
- La UI presenta attività reale, non inventata, ed è significativamente più semplice da usare senza sacrificare funzioni.
- Nessuna funzionalità storica viene eliminata silenziosamente, incluso EvolutionVault; nessun benchmark è spacciato per risultato di general intelligence.
- **Non introdurre progetti linguistici** nel perimetro di Predatore.

**Prima tranche da implementare:** P0, P1 e una prova di fattibilità P2 su singolo hot path misurato. Poi procedere con P3/P4; la UI può essere riorganizzata in parallelo solo se non disturba la verifica dell'engine.
