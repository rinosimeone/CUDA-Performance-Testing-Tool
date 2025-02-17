# CUDA Performance Testing Tool

Un tool completo per il benchmark e l'analisi delle performance CUDA, con visualizzazione grafica dei risultati e confronto con altre GPU NVIDIA.

## Caratteristiche

- ✨ Benchmark completo delle performance CUDA
- 📊 Visualizzazione dei risultati in due modalità:
  - Grafica con matplotlib (salvata in PNG)
  - ASCII-based direttamente nel terminale
- 🔍 Analisi dettagliata comparativa CPU vs GPU
- 📈 Confronto prestazioni con altre GPU NVIDIA
- 💾 Misurazioni della banda di memoria
- 🎯 Raccomandazioni specifiche per il carico di lavoro

## Requisiti

- Python 3.8+
- PyTorch con supporto CUDA
- NVIDIA GPU con driver aggiornati
- matplotlib per visualizzazione risultati
- numpy per elaborazione dati

## Installazione

```bash
git clone https://github.com/yourusername/cuda-test.git
cd cuda-test
pip install -r requirements.txt
```

## Utilizzo

```bash
python test_torch.py
```

## Esempio di Output

Il benchmark fornisce tre tipi di visualizzazione:

1. Confronto CPU vs GPU (ASCII):
```
CPU vs GPU Performance
================================================================================
CPU │ ██████████████████████████████████████████████████ 8.8123s
GPU │ ██████                                             1.1669s
================================================================================
```

2. Confronto con altre GPU NVIDIA (ASCII):
```
GPU Performance Comparison (4000x4000 matrix)
================================================================================
  RTX 4090        │ ██████████                                     0.7234s
  RTX 3090        │ ████████████                                   0.9845s
→ Quadro RTX 3000 │ ██████████████                                1.1669s
  RTX 2080 Ti     │ ███████████████████                           1.3456s
  RTX 2070        │ ████████████████████████                      1.8901s
================================================================================
```

3. Grafici dettagliati (salvati in 'cuda_benchmark_results.png'):
- Confronto diretto CPU vs GPU per diverse dimensioni di matrici
- Grafico del fattore di speedup GPU
- Visualizzazione a barre delle performance comparative

## Risultati Test Recenti

Test eseguito su diverse dimensioni di matrici:

- ✅ Test con matrici 1000x1000 (100 iterazioni)
- ✅ Test con matrici 2000x2000 (50 iterazioni)
- ✅ Test con matrici 4000x4000 (25 iterazioni)
- 📊 Confronto con GPU di riferimento:
  - RTX 4090 (ultima generazione)
  - RTX 3090 (generazione precedente)
  - RTX 2080 Ti e 2070 (due generazioni fa)

## Funzionalità

- Benchmark matriciale con diverse dimensioni
- Confronto diretto CPU vs GPU
- Confronto con altre GPU NVIDIA
- Calcolo dello speedup
- Visualizzazione grafica dei risultati
- Analisi dettagliata delle performance
- Supporto per diverse dimensioni di test

## Changelog

Vedere [CHANGELOG.md](CHANGELOG.md) per la storia completa delle modifiche.

## TODO

- [x] Implementare visualizzazione grafica dei risultati
- [x] Aggiungere benchmark CPU vs GPU
- [x] Implementare calcolo speedup
- [x] Aggiungere confronto con altre GPU NVIDIA
- [ ] Aggiungere supporto per test multi-GPU paralleli
- [ ] Implementare esportazione risultati in vari formati (JSON, CSV)
- [ ] Aggiungere profiling memoria dettagliato
- [ ] Sviluppare modalità batch per test automatizzati
- [ ] Integrare supporto per container Docker

## Contribuire

Le pull request sono benvenute. Per modifiche importanti, aprire prima una issue per discutere le modifiche proposte.

## Licenza

[MIT](https://choosealicense.com/licenses/mit/)
