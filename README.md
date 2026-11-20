# Predatore V7.0

Laboratorio di apprendimento evolutivo logico-binario. EVA apprende durante la caccia e riceve proposte dagli esploratori; SMITH conserva esperienze e sceglie procedure fra una run e l’altra.

**Novità della V7** (dettagli in [RISULTATI_V7_00.md](RISULTATI_V7_00.md)):
- **Nucleo Rust nel training.** Il modulo `predatore_native` esegue episodi interi: mondo, preda, decisione, memorie e feedback. Dà gli stessi risultati del codice Python, verificati bit per bit.
- **Parallelismo.** Esploratori e valutazioni girano su 16 thread di default. Il numero di thread non cambia i risultati.
- **Arena senza simulazioni.** Live mostra il training in corso. Le Registrazioni contengono un episodio di training reale per EVA e per ogni esploratore, a ogni generazione, affiancabili. Il test finale è etichettato TEST.
- **Brain Inspector.** Confronta EVA con la generazione precedente o con l’inizio della run, senza aprire file.
- **Nuova linea sperimentale.** Il generatore degli episodi è xoshiro256** (da Tesseract). I risultati si confrontano con la V6 solo sulle statistiche.

## Avvio su Windows

1. Installa **Python 3.10 o successivo** da python.org (Tkinter è incluso).
2. Estrai l’intero ZIP in una cartella nuova, **fuori da OneDrive**.
3. Facoltativo ma consigliato: installa il nucleo nativo, in uno dei due modi.
   - Esegui **COMPILA_NATIVO.bat**. Serve Rust da https://rustup.rs, con gli strumenti C++ di Visual Studio che l’installer propone.
   - Oppure scarica il pacchetto `.whl` per Windows prodotto dal workflow GitHub `native-wheels` e installalo con `python -m pip install <file>.whl`.
4. Apri **AVVIA_PREDATORE_V7.bat**.

Il titolo della finestra indica il motore in uso: «motore Rust nativo» oppure «motore Python». Senza modulo nativo il laboratorio funziona lo stesso, con gli stessi risultati, ma più lentamente. Verifica da terminale:

```console
python launcher.py --engine-status
```

Su Linux e macOS: `python3 launcher.py --gui`. Il nucleo si compila con `python -m maturin build --release --manifest-path native/Cargo.toml` e il pacchetto si installa con `pip`.

## Prestazioni sul tuo PC

**MISURA_V7.bat** ripete lo stesso training con 1, 4, 8 e 16 thread. Riporta i tempi e verifica che EVA risulti identica in ogni caso. Il report va in `validation/v7/`. Per fissare il numero di thread:

```console
set PREDATORE_WORKERS=12
AVVIA_PREDATORE_V7.bat
```

`PREDATORE_WORKERS=1` esegue tutto in un solo thread; `PREDATORE_NATIVE=0` usa il riferimento Python.

## Equivalenza dei modi di esecuzione

```console
python -m unittest tests.test_v7_native_parity -v
python tools/verify_v7_equivalence.py --generations 5 --runs 3
```

Il primo confronta Python e Rust episodio per episodio sui sette scenari: metriche, eventi di esperienza e cervello esportato. Il secondo confronta l’intero profilo salvato (EVA, storia di SMITH, benchmark, registrazioni) fra Python puro, nativo con un thread e nativo con più thread.

## Avvia un esperimento da terminale

```console
python launcher.py --train --scenario-set extended --seed 83 --smith-profile EVA83 --base-neurons 32 --max-neurons 64 --population 4 --generations 2 --episodes 7 --steps 36 --live
python launcher.py --train --resume --seed 83 --size 8 --generations 3
python launcher.py --smith-status --smith-profile EVA83
python launcher.py --verify EvolutionVault/experiments/EXP_seed83_size8
```

La ripresa rilegge dal checkpoint il profilo e la configurazione della run. I checkpoint V6.06 restano leggibili. Una run V6 ripresa con la V7 continua con il nuovo generatore, come linea V7.

## Confronta le politiche di apprendimento

| Opzione su una nuova run | Funzione |
|---|---|
| `--max-scenario-loss 0` | Conserva il candidato EVA e lo stato precedente. Blocca perdite di catture su ciascuno scenario di validazione e può ripristinare EVA dopo il training personale. |
| `--select-for-learning` | Considera il guadagno degli esploratori su prove di training separate nella selezione. |
| `--relation-credit-horizon 8` | Abilita contatori firmati sulle relazioni e credito distribuito sulle ultime otto decisioni. Usare un nuovo profilo per cambiare orizzonte. |
| `--no-context-distillation`, `--no-circuit-plasticity` | Disattivano i rispettivi meccanismi per un confronto controllato. |
| `--monitor-every 1` | Aggiunge una misurazione di validazione ogni generazione. |

Il bilancio scientifico della V6.06 e i limiti di ciascuna prova sono in **[RISULTATI_V6_06.md](RISULTATI_V6_06.md)**. Le nuove politiche di apprendimento richiedono attivazione esplicita. I dati sul loro rendimento provengono da copie separate e da un numero limitato di linee evolutive.

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

Gli esiti salvati sono in `validation/`. Il prototipo Rust della V6.06 (sonda LUT separata, generatore V6) è conservato in `legacy/v6_lut_probe/` con il suo protocollo in `validation/p2/PROTOCOLLO.md`. Il manuale PDF in `docs/` descrive la V6.05 e le funzioni storiche.
