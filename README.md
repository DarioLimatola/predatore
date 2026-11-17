# Predatore V6.05

Laboratorio di apprendimento evolutivo logico-binario. EVA agisce, apprende e
riceve proposte dai migliori esploratori; SMITH conserva esperienza e strategie
tra run. Il core `boolean_engine` è riusabile senza il simulatore.

## Avvio

Estrarre in una cartella nuova. Serve Python 3.10+ con Tkinter per la GUI.
Windows: `AVVIA_PREDATORE_V6.bat`. Linux/macOS: `python3 launcher.py --gui`.

    python launcher.py --train --seed 42 --smith-profile SMITH
    python launcher.py --train --seed 43 --smith-profile SMITH
    python launcher.py --train --resume --seed 43 --generations 40
    python launcher.py --smith-status

Con seed nuovo e stesso profilo si inizializza una nuova run con memoria
ereditata. Per riprendere un esperimento conservare seed e dimensione. I profili
di versioni precedenti non sono compatibili; conservarli separatamente.

## Documentazione e verifica

- `docs/MANUALE.md` e `docs/Predatore_V6_05_Manuale.pdf`: uso, progettazione, API e limiti.
- `validation/v605/REPORT.md`: risultati, protocollo e riproducibilità.
- `RELEASE_SHA256.json`: impronte dei file della distribuzione.
- `python examples/binary_task.py`: esempio autonomo di feedback binario.
- `python -m unittest discover -s tests -v`: suite di regressione.
- `python tools/benchmark_release.py`: benchmark 2D sequenziale (alcuni minuti).

Le arene non sono state estese al 3D. Non sono stati aggiunti Rust o parallelismo
per il motore. L'archivio su disco `EvolutionVault` serve alla persistenza, ma
non ha una scheda dedicata nella GUI. Il rilascio non contiene dati di run personali.
