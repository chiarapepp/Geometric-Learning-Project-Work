# Note finali per relazione

## Protocollo

Analisi principale su point cloud neuromorfiche in modalita windowed4096: finestre da 4096 eventi, stride 4096, 4096 punti, input [x, y, t] e ordine temporale preservato.

## Copertura

- Combinazioni complete: 23/30.
- Combinazioni mancanti o parziali: 7.

## Risultati principali

- Migliore ricostruzione clean secondo chamfer: nmnist / pointnet_ae / temporal_weighted_chamfer con valore medio 0.00123.
- Loss piu veloce nel benchmark clean: nmnist / hausdorff con 0.002435 secondi medi.
- Migliore convergenza osservata: nmnist / pointnet_ae / temporal_weighted_chamfer alla epoca 49 con val loss 0.001259.
- Modello mediamente migliore sul clean: nmnist / pointnetpp_ae con valore medio 0.004999.
- Ranking complessivo migliore: dvsgesture / pointnetpp_ae / density_aware_chamfer (score 0.052545).

## Lettura consigliata

- Usare `coverage_matrix.md` per dichiarare in modo trasparente quali combinazioni sono complete.
- Usare `loss_benchmark_summary.md` per discutere costo computazionale, memoria e FLOPs stimati.
- Usare `convergence_summary.md` per confrontare velocita e stabilita di addestramento.
- Usare `reconstruction_clean_summary.md` e `corruption_robustness_summary.md` per la qualita finale.
- Usare i plot in `plots/robustness` per mostrare la sensibilita a noise, shuffle temporale e drop.

## Materiale qualitativo

- Pannelli visuali trovati: 0.
- Ablation temporale disponibile: usare `time_weight_ablation_summary.md`.
