# Predatore V6.05
## Manuale del simulatore e del motore booleano vettoriale

Predatore è un laboratorio di apprendimento evolutivo. La rete elabora
osservazioni binarie, sceglie un'azione e accumula evidenza dai suoi esiti. La
popolazione esplora circuiti e rappresentazioni; EVA integra proposte validate
e impara anche durante i propri episodi. SMITH organizza l'esperienza che
attraversa generazioni e run. Il core è indipendente dal simulatore.

## 1. Installazione, nuova run e ripresa

Estrarre lo ZIP in una cartella nuova. Python 3.10 o successivo è sufficiente
per il motore; la GUI richiede Tkinter. Su Windows eseguire AVVIA_PREDATORE_V7.bat,
su Linux/macOS usare il comando sotto. Se Tkinter manca su Linux, installare
il pacchetto della distribuzione per Python e Tk. Il motore gira anche senza GUI.

    python launcher.py --gui
    python launcher.py --train --seed 42 --size 8 --smith-profile SMITH
    python launcher.py --train --seed 43 --size 8 --smith-profile SMITH
    python launcher.py --train --resume --seed 43 --size 8 --generations 40
    python launcher.py --smith-status --smith-profile SMITH

Una run è identificata da seed e dimensione nella radice di lavoro. Per una
nuova run usare una coppia nuova; per la ripresa usare la coppia originale.
Lo stesso nome SMITH eredita EVA e gli specialisti; un nome nuovo crea una
linea indipendente. La ripresa usa la configurazione salvata e permette di
aumentare il numero di generazioni. L'arresto da GUI avviene al confine
della generazione. Una run vecchia non può riportare indietro un profilo
già avanzato con un'altra run.

Non si importano automaticamente memorie V6.04 o precedenti. Formati del
rilascio: boolean-vector/5, smith.supervisor/2, predatore.evolution-vault/3.0.
Tenere le vecchie versioni in cartelle separate.

## 2. I ruoli e i confini del software

- Esploratori: agenti adattivi con circuiti e memorie distinti. Sono valutati
  su tutti e quattro gli scenari di training. I migliori e gli specialisti
  diventano docenti; i discendenti ereditano memorie di evidenze e circuito
  dal proprio genitore, prima di mutazione e distillazione.
- EVA - learning agent: agente principale. Può adottare conoscenza
  contestuale, un docente intero o una modifica chirurgica di LUT. Continua
  a imparare anche quando nessuna proposta viene accettata. Le copie di
  valutazione sono effimere, non un'EVA congelata persistente.
- SMITH - agent supervisor: non è un attore dell'arena. Conserva profilo,
  specialisti, campione di eventi, riepiloghi, compiti di validazione e
  statistiche delle strategie. La sua scelta è fra procedure implementate,
  non un generatore autonomo di nuovi algoritmi.

I file engine/continual.py, engine/smith.py e engine/challenges.py gestiscono
generazioni, supervisione e curriculum. engine/world.py è il simulatore;
engine/evaluation.py coordina gli episodi. La GUI visualizza salvataggi e
telemetria. Il pacchetto boolean_engine non importa mondo, GUI o filesystem:
network.py calcola la rete, tensor.py implementa gli assi, context.py gli
indirizzi, memory.py le evidenze, hdc.py le associazioni e distillation.py
le proposte dei docenti. I moduli di importazione sotto engine/ puntano allo
stesso core: non esiste un selettore legacy.

## 3. Che cosa significano moduli, bit e assi

Un agente è un grafo di moduli, un decoder e memorie. Lo stato di ogni modulo è
un intero Python con W bit. L'indice 0 è il bit meno significativo. Gli assi
interpretano quegli stessi bit in ordine per righe: l'ultimo asse varia più
rapidamente. Un vettore da 64 bit può essere (64,), (8,8) o (4,4,4); non sono
64, 8 o 4 neuroni, ma forme diverse di un singolo modulo. La larghezza HDC è
una quantità separata.

Per due ingressi A e B, ogni corsia usa una delle quattro maschere LUT:

    F = (~A & ~B & T00) | (~A & B & T01)
      | ( A & ~B & T10) | ( A & B & T11)
    write = write_mask & ~hold & width_mask
    next = ((old & ~write) | (F & write)) & width_mask

La funzione scalare iniziale AND, OR, XOR, COPY e simili serve a creare le
LUT. Una mutazione può differenziare singole corsie. Il collegamento hold
preserva le corsie protette; la maschera di scrittura preserva le altre
corsie disabilitate. Tutti i moduli leggono i vettori del tick precedente:
si evitano dipendenze dall'ordine di esecuzione. Un ritardo d legge gli
ingressi di d tick prima; prima che esista la storia necessaria legge zero.
La ricorrenza è iterativa e limitata, con massimo 8 tick per decisione
e 3 tick di ritardo. Il reset cancella lo stato transitorio dell'episodio,
non il circuito né le memorie acquisite.

I sensori sono codificati usando rotazioni del pacchetto di bit e il sensore
replicato via XOR (oppure broadcast nel core). Per ogni bit di azione il
circuito legge una corsia di un modulo di uscita. Il mondo interpreta il
codice risultante.

### Calcolo sparso

L'esecuzione segue gli antenati strutturali delle uscite, attraversando A, B e
hold. Visita finita con insieme dei nodi già incontrati: anche un ciclo termina.
I moduli dormienti restano nel genoma, ma non si calcolano finché non vengono
connessi. La cache del sottografo è invalidata dalle mutazioni al circuito. Lo
stato dei nodi esclusi resta zero. Il conteggio degli antenati è prudente: una
LUT costante può rendere ininfluente una sorgente collegata. «Eseguito» non
significa «causalmente indispensabile».

### Assi effettivi

rotate gira il vettore intero; axis_roll scorre un asse preservando le altre
coordinate; neighbor_or e neighbor_xor combinano una corsia e i suoi vicini
ciclici; transpose permuta l'asse selezionato con il successivo. La forma è
locale alla regola e mantiene W. Con sole rotazioni piatte, cambiare forma
non produce vantaggi. Con operatori per asse, può cambiare i risultati.

Le mutazioni propongono operatore, asse, ritardo, nuova forma (divisione,
fusione, scambio), modifica della storia, della maschera HDC o espansione della
larghezza. Al raddoppio, le LUT preservano i bit esistenti e non inventano
conoscenza nei bit nuovi. La selezione misura i candidati, non calcola una
forma ottima generale. Le primitive tensoriali sono testate; il simulatore
rimane 2D.

## 4. Memorie e decoder

L'indirizzo breve contiene osservazione corrente, precedente e validità del
passato. L'indirizzo contestuale contiene il presente e, dal più recente,
coppie osservazione/azione passate. Con S sensori, A bit di azione e H coppie,
usa S + H*(S+A) bit di dati più H di validità. Valori predefiniti 15, 3 e 3:
72 bit. Le chiavi non usano un hash riduttivo; due storie osservate uguali
in situazioni fisiche diverse restano comunque ambigue.

Per ogni azione, un contatore positivo e uno negativo saturano a 65535.
Il piano j contiene il bit j di tutti i contatori, un bit per azione.
Una somma usa XOR e riporti booleani in parallelo fra corsie; un confronto
esamina i piani dal più significativo. Il decoder cerca azioni con almeno
due evidenze e margine positivo. A parità, preferisce l'azione proposta e
poi l'ordine circolare. Una proposta con supporto negativo può essere
sostituita da un'azione non smentita: non significa che questa sia buona.

L'ordine di decisione è: circuito, eventuale RAM breve, eventuale RAM
contestuale, altrimenti recupero HDC se abilitato. Il decoder annota l'origine
finale; una memoria può confermare la stessa azione del circuito. Contare una
decisione da HDC non equivale a dimostrare che HDC ha causato una cattura.

### Canali, feedback e oblio

Il segno del reward aggiorna la RAM breve e le evidenze contestuali personali.
Una cattura accredita con peso 3 i contesti della traccia recente, nel canale
terminale. Questo è credito euristico: non prova che ogni azione precedente
sia stata indispensabile. Il terzo canale riceve testimonianze dei docenti;
è separato, anche quando il decoder somma i tre canali.

Ogni 1024 aggiornamenti di un banco, il successivo accesso applica il
dimezzamento delle evidenze del contesto secondo gli intervalli trascorsi.
I nomi dei docenti sono conservati fino a otto per contesto. Le memorie sono
limitate per numero di righe, con espulsione fra le più vecchie e preferenza
per tracce terminali e provenienza. Accorciare la storia può fondere righe;
allungarla non ricrea osservazioni assenti. La memoria predittiva registra
l'ultimo esito sensoriale per indirizzo e azione: la differenza XOR è
diagnostica e non guida la pianificazione.

### Recupero associativo HDC/VSA

L'encoding associa ruolo e valore tramite XOR, aggrega con maggioranza per
corsia e distingue i lag con permutazioni. La distanza di Hamming è il numero
di bit diversi. I prototipi sono vettori da 512 bit, fino a 256 per rete.
Si cercano al massimo 16 candidati recenti, ancorati alla stessa osservazione
corrente sotto la maschera di attenzione; fino a quattro vicini con distanza
non superiore a un quarto della larghezza. Sono richiesti almeno due contesti
distinti concordi e un vantaggio netto sull'alternativa; altrimenti il sistema
si astiene. Il veto di evidenze negative esatte rimane valido.

La maschera iniziale considera tutti i sensori; l'evoluzione può modificarla.
La vicinanza nell'encoding non è equivalenza di significato. I prototipi
possono confondersi; due vicini non sono due prove statisticamente indipendenti.
Una parola convertita in un ID binario non acquista da sola semantica.

## 5. Una generazione e la promozione

1. Generare le mappe di training, seed e programma condiviso. Sostituire fino
   a circa un quarto delle condizioni con replay da sole run di training.
2. Far apprendere ogni esploratore negli stessi episodi; registrare un
   campione degli eventi e selezionare docenti complessivi e per scenario.
3. Costruire proposte di distillazione contestuale, docenti interi e
   cambiamenti locali di circuito. Valutarle su copie adattive isolate.
4. Accettare un candidato competitivo e poi eseguire il training personale
   di EVA. Misurare separatamente la regressione o il guadagno post-training.
5. Aggiornare strategie e archivi SMITH. Generare discendenti, salvare
   checkpoint locali e globali e pubblicare metriche e circuito della generazione.

Distillazione: deduplicazione per ID o impronta; aggregazione per contesto
e sorgente prima del voto globale. Mille eventi dello stesso agente non valgono
mille docenti. Un nuovo conflitto può ritirare una testimonianza precedente
senza cancellare esperienza personale. Il limite è 128 contesti per chiamata.
L'adozione di un docente intero trasferisce anche la sua memoria; non è
un'unione lossless dei due agenti.

Plasticità del circuito: durante il training una traccia limitata registra
i pattern LUT usati. Le tracce in cui il decoder ha sostituito la proposta
non vengono attribuite al circuito. Evidenze locali a quattro bit possono
proporre il ribaltamento di un singolo bit LUT. Le proposte vengono valutate:
non si modifica automaticamente il circuito dopo ogni errore.

Novità: cattura, esplorazione e stalli per scenario descrivono il comportamento.
La distanza media dai vicini aiuta a scegliere genitori non ridondanti.
Crescita e pruning non hanno timer: prima si riusa capacità dormiente, poi
si tenta crescita nei limiti. Le mutazioni dannose non hanno accesso diretto a EVA.

Il punteggio di promozione è lessicografico: catture bilanciate, catture nel
compito corrente, stalli bilanciati, reward e minor numero di moduli. Corrente
e insieme dei compiti precedenti pesano ciascuno metà quando entrambi esistono.
Catture e stalli sono frazioni esatte, non confronti con un epsilon arbitrario.
Sono possibili perdite per scenario compensate da altri vantaggi. Gli esiti
appaiati nei report rendono visibile quali episodi hanno cambiato risultato.

## 6. SMITH, persistenza e valutazione

SMITH alterna consolidate (credito terminale nella distillazione), explore
(genitori più nuovi), represent (più cambiamenti di rappresentazione) e
compact (potatura dormienti). Il contesto ha sei bit: poche catture, molti
stalli, trasferimento respinto, conflitto, sottoutilizzo e regressione
post-training. Il supervisore contabilizza i tentativi e i miglioramenti
di validazione per contesto; mantiene esplorazione periodica. Non può
isolare tutte le cause di un risultato o garantire la strategia migliore.

Il profilo mantiene fino a 16 condizioni di replay, 4 specialisti, 64
riepiloghi di run e 512 generazioni di storia, più i budget di configurazione.
Gli eventi di tutti gli attori vengono campionati; EVA viene istruita
dai docenti selezionati, non da ogni singolo evento conservato.

Addestramento, validazione e holdout usano condizioni e seed separati.
Le copie di valutazione partono dal candidato salvato, azzerano lo stato
ricorrente a ogni episodio e possono apprendere al loro interno; vengono
scartate senza riportare transizioni nel profilo. L'holdout serve solo
a confronto finale, replay e diagnostica. I test iniziale/finale/pop-medio
usano le stesse condizioni. La salita del training da sola non dimostra
generalizzazione perché le mappe cambiano fra generazioni.

EVA può fare peggio della media: è una singola politica, una distillazione
può perdere specializzazioni, il reward locale non coincide con le catture
e le condizioni del prossimo test sono diverse. La continuità è verificabile
tramite identità del cervello ereditato; la superiorità non è garantita.

## 7. Il mondo e la GUI

I quattro scenari sono preda ferma, preda mobile, più ostacoli e visibilità
ridotta. Il sensore non espone coordinate assolute nascoste. L'osservazione
contiene visibilità, direzione visibile N/S/O/E, quattro celle libere a
distanza 1 e a distanza 2, preda visibile vicina e collisione già avvenuta
nell'episodio. Le azioni 0-3 sono passi cardinali, 4-7 salti cardinali
di due celle sopra un ostacolo con atterraggio libero.

Reward: cattura +20, mossa illegale -1, cella nuova +0,12, cella già vista
-0,04 e costo ulteriore del salto 0,12. Venti posizioni consecutive in non
più di due celle fermano l'episodio e sottraggono 2 alla metrica finale, non
al feedback di una nuova transizione. Il riferimento greedy vede la posizione
della preda; il random sceglie mosse cardinali. Non rappresentano agenti con
le medesime informazioni o lo stesso decoder di EVA.

In Addestramento: configurare seed, profilo, popolazione, passi e generazioni.
Default min/base/max 32/64/128, forma iniziale 8×8, massimo 256 bit/modulo,
2048 indirizzi per banco e 2048 eventi condivisi. Cambiare il massimo altera
il budget di mutazione, non forza il numero di moduli utilizzati. Minimo,
base e massimo sono interi con slider e attivazione personalizzata. La base
inizializza reti nuove; soglie di un profilo ereditato devono restare compatibili
con larghezza e parametri memoria/ricorrenza già salvati.

Live mostra episodi realmente eseguiti, Preview una simulazione indipendente,
Replay episodi salvati. Nel disegno blu e verde indicano il bit zero; in
telemetria il grigio indica moduli non eseguiti. Il clic mostra vettori e
decodifica; «Confronta con cervello JSON» accetta un cervello o uno snapshot
di circuito per confrontare nodi, regole, forme e uscite. I grafici usano blu
per EVA e verde per la popolazione, generazioni reali sull'asse orizzontale,
neuroni interi e passi medi calcolati sulle catture riuscite.

Non c'è una scheda Evolution Vault. L'archivio su disco resta necessario.
Il monitor aggiunge misure senza insegnare agli agenti. La diagnostica
usa copie isolate, incluse ablation HDC, operatori per asse, ricorrenza e
memoria. La disattivazione degli operatori per asse lascia le rotazioni piatte.
EVA e SMITH sono etichettati separatamente.

## 8. Dati, costi e verifica

Cartelle principali sotto EvolutionVault/experiments/EXP_seed..._size...:

- config.json e manifest.json: parametri e protocollo.
- environment/suites e environment/training: suite fisse e schedule di training.
- checkpoints/latest.json: popolazione, profilo SMITH ed esperimento.
- brains/initial_eva.json e brains/eva.json: stato iniziale e ultimo.
- evolution/: metriche, promozioni, contributi e mutazioni.
- benchmarks/: EVA iniziale/finale, popolazione, riferimenti random e greedy.
- neuroscience/generations/: ultimi 256 snapshot di circuito, senza RAM.
- replays/, live/ e monitor/: osservazione, copie e statistiche.

Il profilo globale è EvolutionVault/SMITH/<profilo>. Si scrive su un file
temporaneo, poi si sostituisce il checkpoint; se Windows blocca il file,
il programma può creare un fallback completo. Un lock evita scrittori
simultanei sullo stesso profilo. Il checkpoint locale viene prima di quello
globale per consentire recupero; non spostare i dati mentre si addestra.

Il costo di inferenza cresce con moduli raggiungibili, tick e larghezza,
oltre alle ricerche e agli aggiornamenti di memoria. Le operazioni su interi
Python grandi non sono singole istruzioni CPU per una rete intera. Il
massimo congiunto è 262144 bit di stato dei moduli: non si possono combinare
tutti i massimi individuali. I casi controllati sono nel report del rilascio.
Non si può inferire il fattore di accelerazione dell'intero training da un
microbenchmark inferenziale; valutazione, copie e salvataggi hanno costi propri.

## 9. API per riusare il motore

Copiare boolean_engine/ oppure installare il solo core con python -m pip install .
L'applicazione deve definire quali bit osserva, il significato delle azioni,
la funzione di reward e i confini temporali. Il core non vede coordinate,
identificatori di mappa o informazioni future.

    from boolean_engine import BinaryVectorNetwork
    net = BinaryVectorNetwork(seed=19, sensors=2, action_bits=1,
                              neurons=4, shape=(8,), context_history=1)
    net.reset()
    action = net.decide((1, 0))
    net.transition((0, 1), captured=False)
    net.feedback(1)

Usare nell'ordine decide, transition e feedback una sola volta per passo.
captured=True indica un successo terminale del dominio. Con action_bits=b
l'intero restituito è compreso fra 0 e 2**b-1. reset() conserva l'apprendimento.
export() restituisce un dizionario JSON; restore() ricrea rete e runtime.
snapshot_id() identifica struttura e memorie escludendo lo stato di episodio.
Le opzioni diagnostiche non sono parte dello snapshot: riapplicarle alle copie.

Costruttore: sensori 1-1024, azioni da 1-8 bit, moduli 1-4096 per una rete
eseguibile, forma 1-8 assi e larghezza 8-4096, tick 1-8, traccia 1-64,
storia 0-8. La capacità indirizzi è impostabile (default 2048); quella
HDC non supera il budget memoria e la larghezza HDC è impostabile 128-4096.
Per cambiare la struttura usare grow, remove_neuron, vary_topology,
mutate_representation, reshape o extend_width. Se si interviene direttamente
sulle connessioni in un esperimento, usare invalidate() e reset().

Per produrre esperienze impostare collect_experience e experience_sink.
Un evento contiene observation impacchettato, history dal più vecchio al più
recente, action, feedback (0 neutro/1 positivo/2 negativo), terminal_success,
eventuali eligible_contexts e next. Aggiungere source ed event_id stabili a
livello di applicazione. distill_into(student, events) aggiorna la memoria
docente e resetta lo stato transitorio; lavorare su una copia per validare.
Non mischiare eventi di domini con diversi significati dei sensori o azioni.

L'esempio examples/binary_task.py mostra apprendimento di una tabella XOR
da feedback. Illustra il contratto API e la memoria, non un'inferenza
linguistica. Il prossimo programma può riusare il core e sostituire il
dominio. Parole come vettori sono possibili come codifica, ma apprendere
significati e generare linguaggio richiede progettazione, dati e prove dedicate.
