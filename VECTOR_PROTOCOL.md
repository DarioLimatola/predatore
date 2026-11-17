# Protocollo V6.05

Circuito: boolean-vector/5. Profilo: smith.supervisor/2. Archivio:
predatore.evolution-vault/3.0. Una sola implementazione di inferenza.

Il training cambia deterministicamente tra generazioni. Solo condizioni di
training già eseguite sono eleggibili al replay. EVA ed esploratori ricevono
gli stessi episodi entro una generazione. Le proposte vengono confrontate
in validazione su copie adattive isolate; l'holdout non influenza il training.
Il ciclo dei ruoli, il decoder e le regole di promozione sono descritti in
docs/MANUALE.md. I report sono in validation/v605/.
