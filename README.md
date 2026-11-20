# Predatore V6.06

Laboratorio di apprendimento evolutivo logico-binario. EVA apprende durante la caccia e riceve proposte dagli esploratori. SMITH conserva esperienze e sceglie procedure fra una run e l’altra. La configurazione iniziale usa il circuito Python e i quattro scenari V6.05.

## Avvia il laboratorio

Estrai l’intero ZIP in una cartella nuova. Su Windows apri **AVVIA_PREDATORE_V6.bat**. Su Linux e macOS esegui `python3 launcher.py --gui` dalla cartella `predatore`. Servono Python 3.10 o successivo e Tkinter. Il training da terminale usa la libreria standard. Rust è opzionale per riprodurre il confronto nativo.

Per una prima run sui **sette scenari** con visualizzazione dei frame reali, scegli il curriculum esteso nella GUI oppure esegui:

```console
python launcher.py --train --scenario-set extended --seed 83 --smith-profile EVA83 --base-neurons 32 --max-neurons 64 --population 4 --generations 2 --episodes 7 --steps 36 --live
python launcher.py --train --resume --seed 83 --size 8 --generations 3
python launcher.py --smith-status --smith-profile EVA83
python launcher.py --verify EvolutionVault/experiments/EXP_seed83_size8
```

La ripresa rilegge dal checkpoint il profilo e la configurazione della run. Nel laboratorio scegli lo stesso esperimento per osservare Live Training, una registrazione o una simulazione indipendente. Il curriculum esteso aggiunge un labirinto a corridoi, una prova a premio ritardato e una preda con una propria rete. Il curriculum classico rimane disponibile per confrontare i risultati storici.

## Confronta le politiche di apprendimento

| Opzione su una nuova run | Funzione |
|---|---|
| `--max-scenario-loss 0` | Conserva il candidato EVA e lo stato precedente. Blocca perdite di catture su ciascuno scenario di validazione e può ripristinare EVA dopo il training personale. |
| `--select-for-learning` | Considera il guadagno degli esploratori su prove di training separate nella selezione. |
| `--relation-credit-horizon 8` | Abilita contatori firmati sulle relazioni e credito distribuito sulle ultime otto decisioni. Usare un nuovo profilo per cambiare orizzonte. |
| `--no-context-distillation`, `--no-circuit-plasticity` | Disattivano i rispettivi meccanismi per un confronto controllato. |
| `--monitor-every 1` | Aggiunge una misurazione di validazione ogni generazione. |

Il bilancio scientifico e i limiti di ciascuna prova sono in **[RISULTATI_V6_06.md](RISULTATI_V6_06.md)**. Le nuove politiche di apprendimento richiedono attivazione esplicita. I dati sul loro rendimento provengono da copie separate e da un numero limitato di linee evolutive.

## Riprendi un esperimento V6.05

Scegli la cartella della singola run originale e una cartella di destinazione ancora inesistente. Questo esempio usa il seed 42 e mappe 8×8. Sostituiscili con quelli indicati nel tuo `config.json`. Il numero di generazioni richiesto deve essere almeno quello già completato.

```console
python launcher.py --fork-v605 "../Predatore605/EvolutionVault/experiments/EXP_seed42_size8" --vault-root "../Linea606"
python launcher.py --train --resume --vault-root "../Linea606" --seed 42 --size 8 --generations 40
python launcher.py --gui --vault-root "../Linea606"
```

Il fork conserva il profilo SMITH della transazione salvata nella run scelta. Registra l’impronta del checkpoint d’origine nel manifest. Per il rollback riapri la cartella originale con V6.05. La revisione di riferimento è `8f724235accc041c9e11c700786ac55cec9e3123` di `DarioLimatola/predatore`.

## Riproduci le verifiche

```console
python -m unittest discover -s tests -v
python examples/binary_task.py
python tools/benchmark_release.py
python -m tools.benchmark_worlds --output Analysis/worlds.json
python -m tools.candidate_exchange --help
```

Gli esiti salvati sono in `validation/`. Le istruzioni per compilare il prototipo Rust sono in `validation/p2/PROTOCOLLO.md`. Il manuale PDF in `docs/` descrive la V6.05 e le funzioni storiche.
