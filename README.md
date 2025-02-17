# CUDA Performance Testing Tool

Un tool completo per il benchmark e l'analisi delle performance CUDA, con visualizzazione grafica dei risultati.

## Caratteristiche

- ✨ Benchmark completo delle performance CUDA
- 📊 Visualizzazione dei risultati in due modalità:
  - Grafica con matplotlib (salvata in PNG)
  - ASCII-based direttamente nel terminale
- 🔍 Analisi dettagliata comparativa CPU vs GPU
- 📈 Analisi dello scaling delle performance
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

Il benchmark fornisce due tipi di visualizzazione:

1. ASCII nel terminale:
```
Performance Visualization
================================================================================
CPU │ ██████████████████████████████████████████████████ 8.8123s
GPU │ ██████                                             1.1669s
================================================================================
```

2. Grafici dettagliati (salvati in 'cuda_benchmark_results.png'):
- Confronto diretto CPU vs GPU per diverse dimensioni di matrici
- Grafico del fattore di speedup GPU
- Visualizzazione a barre delle performance comparative

Output testuale include:
```
=== Running CUDA Benchmarks ===
Matrix Multiplication (1000x1000, 100 iterations)
CPU Time: X.XXX seconds
GPU Time: X.XXX seconds
Speedup: XX.XX times faster

[Altri risultati per dimensioni maggiori...]
```

## Risultati Test Recenti

Test eseguito su diverse dimensioni di matrici:

- ✅ Test con matrici 1000x1000 (100 iterazioni)
- ✅ Test con matrici 2000x2000 (50 iterazioni)
- ✅ Test con matrici 4000x4000 (25 iterazioni)
- 📊 Risultati visualizzati in grafici comparativi

## Funzionalità

- Benchmark matriciale con diverse dimensioni
- Confronto diretto CPU vs GPU
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
- [ ] Aggiungere supporto per test multi-GPU paralleli
- [ ] Implementare esportazione risultati in vari formati (JSON, CSV)
- [ ] Aggiungere profiling memoria dettagliato
- [ ] Sviluppare modalità batch per test automatizzati
- [ ] Integrare supporto per container Docker

## Contribuire

Le pull request sono benvenute. Per modifiche importanti, aprire prima una issue per discutere le modifiche proposte.

## Licenza

[MIT](https://choosealicense.com/licenses/mit/)
