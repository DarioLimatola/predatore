# Credito temporale e collegamenti plastici

## Comportamento disponibile

Una linea sperimentale può apprendere collegamenti dai moduli attivi ai bit dell'azione. L'opzione `--relation-credit-horizon 8` conserva le ultime otto decisioni per il feedback. Con orizzonte 1 contribuisce solo la decisione corrente. Il valore predefinito 0 conserva il circuito e i decoder V6.05.

Il circuito fornisce al massimo 16 moduli antenati delle uscite, ordinati per indice. Da ciascuno si legge la lane `indice_modulo % larghezza`. Per ogni combinazione di modulo, valore osservato e bit motore esiste un contatore indipendente. Un premio sostiene il bit scelto e una penalità sostiene il suo complemento. Le osservazioni in conflitto si compensano. Una singola ricompensa può aggiungere al massimo un voto per collegamento, anche se l'azione si è ripetuta a lungo. Le decisioni prese dalla memoria fanno avanzare l'orologio e non ricevono attribuzione diretta per il circuito.

Con almeno due voti dello stesso segno il collegamento partecipa alla decisione. Il segno della maggioranza sceglie il bit e la parità conserva la proposta della LUT. I decoder contestuali rimangono disponibili dopo questo passaggio. Gli aggiornamenti strutturali lasciano intatte le LUT. La loro larghezza numerica passa da 16 a 32 e a 64 bit se necessario e satura all'estremo di 64 bit. In Python ogni contatore è un oggetto e occupa memoria aggiuntiva oltre alla sua precisione numerica.

Il reset di episodio cancella la traccia e mantiene i contatori. Le mutazioni che cambiano il significato dei segnali invalidano i collegamenti appresi. Il genoma sperimentale usa il codec `boolean-vector/5+relations/1`, anche per il salvataggio a metà episodio. Il codec storico resta `boolean-vector/5`. Le copie con plasticità disattivata leggono i pesi esistenti. Il frame mostra fino a otto modifiche recenti indicando esplicitamente l'eventuale troncamento.

## Risultato sulle mappe di trasferimento

`python tools/benchmark_relation_credit.py` riproduce l'esperimento dalla radice del progetto e scrive `validation/p3/relations.json`. Per ogni seed del circuito (42, 117, 809) e scenario sono state generate 8 mappe di training 8×8 e 16 mappe di prova 10×10. Ogni episodio ha un massimo di 48 passi. Tutte le varianti per una stessa prova condividono inizializzazione, mappa e seed dell'ambiente. Ogni scenario addestra un agente dedicato con 32 moduli da 64 bit. La prova usa una copia congelata e le catture riportate sono complessive su 48 episodi per colonna e scenario.

| Scenario | Circuito di riferimento | Proposta locale a un bit LUT | Credito corrente | Credito a 8 passi | Stesso agente a 8 passi, collegamenti esclusi in prova |
|---|---:|---:|---:|---:|---:|
| Static | 1 | 1 | 0 | 1 | 0 |
| Hidden | 0 | 0 | 2 | 0 | 0 |
| Maze con corridoi | 1 | 1 | 2 | 1 | 1 |
| Premio differito | 0 | 0 | 0 | 0 | 0 |

La variante LUT raccoglie l'evidenza locale per lane e applica una sola mutazione proposta al termine del training, se disponibile. Il controllo con collegamenti esclusi conserva le memorie apprese dallo stesso individuo. I dati per seed, gli episodi individuali, i tempi per variante, le decisioni per decoder e i bit mutati sono nel JSON.

**Gate P3 non superato.** Il criterio fissato nel programma richiedeva più catture rispetto al riferimento e all'orizzonte 1 sia nel labirinto sia nel premio differito, con un vantaggio su almeno due seed per ciascun confronto e senza perdite aggregate negli altri compiti. Nel labirinto le differenze a 8 passi rispetto al riferimento sono −1, +1, 0 sui tre seed. Nel compito differito il gruppo a 8 passi ha ottenuto zero catture anche nei 24 episodi di training. In quel gruppo manca quindi il segnale positivo necessario a distinguere la qualità delle regole di credito. Nel complesso il nuovo decoder ha determinato 2.470 azioni nei probe a 8 passi. La regola è operativa e resta sperimentale.

Una prova controllata costruisce un circuito che entra in contatto al primo passo e continua a muoversi verso sud. Nell'ambiente reale con conferma dopo quattro ulteriori decisioni, il test verifica il feedback [0, 0, 0, 0, 20]. Dopo due episodi il segnale «preda a sud», presente al passo del contatto, ha due voti con orizzonte 8 e zero con orizzonte 1. Questo controlla la propagazione temporale con opportunità di apprendimento garantita. La rilevanza causale dei moduli interni, il credito profondo multilivello e la generalizzazione a nuove mappe restano questioni aperte.

I test si eseguono con `python -m unittest discover -s tests -p "test_relation_credit.py"`.
