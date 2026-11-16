# V6.04 — protocollo operativo

## Controllore booleano

Il circuito usa moduli ricorrenti di bit, LUT vettoriali, maschere di scrittura,
rotazioni e uscite di tre bit per otto azioni. La RAM breve usa osservazione
corrente/precedente/validità; la RAM contestuale usa tre coppie storiche
osservazione/azione più osservazione e validità correnti, 72 bit in totale.
Le chiavi non contengono coordinate, nomi degli scenari o stati neurali privati.

Ogni contesto conserva contatori positivi/negativi per azione su quattro piani
di otto bit. Incremento saturante, confronto e sottrazione usano logica booleana.
La scelta richiede almeno due evidenze e prevalenza positiva; evidenze negative
sufficienti possono respingere la proposta corrente. In assenza di consenso
resta la decisione del decoder precedente. Le catture attribuiscono credito a
una traccia recente limitata: questa è un’euristica, non una prova di causalità.

Reset: vettori, storia e tracce temporanee. Persistenza personale: circuito,
RAM breve/contestuale e predittore. Le memorie hanno capacità finita e possono
espellere righe. La forma n-dimensionale non modifica ancora gli operatori per asse.

## Proprietà dei dati e ruoli

Il checkpoint SMITH contiene `eva` (un singolo cervello), `events`, `explorers`,
`tasks`, `teacher_sources`, `novelty`, `contributors`, `meta` e la storia delle
misure. Risiede in `EvolutionVault/SMITH/<profilo>`. `meta` contiene statistiche
contestuali e riepiloghi delle run; non contiene un altro predatore.

Il checkpoint dell’esperimento contiene `supervisor_state`, la popolazione per
la prossima generazione e `evaluated_population`, con gli esploratori appena
valutati. L’anteprima mostra questi ultimi con la generazione corretta.
`brains/eva.json` è la sola copia esportata dell’agente corrente dell’esperimento.

Ogni evento di training riceve un ID run/generazione/agente/transizione.
SMITH campiona ogni esploratore ed EVA nel proprio archivio limitato. Prima
dell’espulsione del reservoir, i campioni degli insegnanti correnti sono comunque
considerati per il trasferimento. I precedenti insegnanti restano eleggibili
finché le loro esperienze sono nell’archivio. I feedback dei test non sono raccolti.

## Trasferimento e apprendimento

Il leader globale e gli specialisti dei quattro scenari vengono scelti dai
risultati di training, fino al budget di tre insegnanti. La distillazione aggrega
un’opinione per fonte/contesto/azione, con credito terminale limitato e più forte.
Le righe sono ricostruite dalle evidenze, senza accumulare falsi voti a ogni replay.
Contatori correlati non sono una misura statistica di indipendenza.

Le proposte sono: EVA distillata, miglior insegnante distillato, insegnanti
intatti e una modifica locale di LUT guidata dal feedback. Sono deduplicate per
contenuto completo del circuito e delle memorie. Il massimo di 128 scritture
contestuali si applica a ogni candidata; i contatori nel report distinguono
la candidata costruita su EVA dal trasferimento effettivamente selezionato.
`contexts_consolidated` conta le scritture della candidata scelta, prima degli
ulteriori aggiornamenti della vita di EVA; non conta contesti nuovi unici o
conoscenze necessariamente preservate per sempre.

Il riferimento è EVA corrente. Il primo criterio è metà catture della run
corrente e metà media delle catture dei compiti precedenti (o solo corrente
quando manca il passato). Spareggi: catture correnti, meno stalli bilanciati,
reward bilanciato, meno neuroni. Catture e stalli usano rapporti razionali da
conteggi interi. In caso di parità completa può passare una memoria diversa.
Le perdite per scenario e gli esiti appaiati guadagnati/persi restano nei log.
Questo ranking non è un test di significatività e può ammettere dimenticanza.

Le prove procedono su copie adattive separate per compito; apprendono fra gli
episodi del compito ma non modificano i candidati. Il risultato viene memorizzato
in una cache limitata, identificata dal cervello iniziale completo e dal compito.
Il training reale non usa questa cache. Le misure finali usano lo stesso protocollo
adattivo per EVA e popolazione, su mappe e seed riservati.

Dopo la selezione EVA esegue sempre il training con plasticità attiva. Le misure
`capture_rate` e `population_mean_capture` vengono dai rispettivi episodi reali,
con identiche mappe/seed/budget. `eva_lifetime_changed` e `eva_updated` distinguono
aggiornamento personale da trasferimento. I risultati delle prove finali non
rientrano in archivio, insegnamento, riproduzione o strategie SMITH.

## Struttura, supervisione e recupero

Base 64, minimo 32, massimo 128 predefiniti. Nella discendenza mutata la dimensione
può variare di −1, 0, +1, simmetricamente prima dell’applicazione dei limiti.
La rimozione fisica rimappa riferimenti e uscite; le aggiunte vengono collegate
all’uscita. Non esistono contatori di età per crescita/potatura. Una topologia
aumentata o ridotta arriva a EVA solo attraverso la proposta selezionata, salvo
l’adeguamento esplicito a nuovi limiti di risorse fra run.

SMITH apprende due strategie con statistiche globali e per contesto a quattro
bit: catture basse, stalli alti, trasferimento respinto, conflitti. `consolidate`
privilegia evidenze terminali; `explore` considera anche altri feedback e usa
più genitori scelti per novità. La ricompensa è il miglioramento del punteggio
di validazione, attribuito alla strategia che ha prodotto la generazione.
Il numero di neuroni può variare in entrambe le strategie.

Commit locale prima del profilo SMITH; lock a singolo scrittore; revisione e
transazione impediscono riprese storiche che sovrascrivano run successive.
La ripresa ripara una pubblicazione globale interrotta e può ricostruire i
risultati finali da un ultimo checkpoint già completo. Monitor e osservatore
non influenzano i risultati del training. Il pacchetto contiene un solo albero
`predatore/` e nessun archivio runtime o motore delle altre versioni.
