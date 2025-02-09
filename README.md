# CUDA Performance Testing Tool

Un tool completo per il benchmark e l'analisi delle performance CUDA, con visualizzazione ASCII dei risultati.

## Caratteristiche

- ✨ Benchmark completo delle performance CUDA
- 📊 Visualizzazione ASCII dei risultati senza dipendenze esterne
- 🔍 Analisi dettagliata comparativa con altre GPU
- 📈 Analisi dello scaling delle performance
- 💾 Misurazioni della banda di memoria
- 🎯 Raccomandazioni specifiche per il carico di lavoro

## Requisiti

- Python 3.8+
- PyTorch con supporto CUDA
- NVIDIA GPU con driver aggiornati

## Installazione

```bash
git clone https://github.com/yourusername/cuda-test.git
cd cuda-test
pip install -r requirements.txt
```

## Utilizzo

```bash
python check_cuda.py
```

## Esempio di Output

```
Performance Moltiplicazione Matrice 1024x1024 (ms)
============================================================================
Current     │ █████████████████                                  0.8
RTX 4090    │ ███████████████████████████                        1.2
RTX 3090    │ ██████████████████████████████████                 1.5
```

## Risultati Test Recenti

Test eseguito su Quadro RTX 3000:

- ✅ Eccellenti performance su matrici piccole (1024x1024): 0.8ms
- ✅ Performance competitive su matrici medie (2048x2048): 5.6ms
- ℹ️ Performance nella media per matrici grandi (4096x4096): 36ms
- ℹ️ Banda memoria: 256 GB/s (nella media delle GPU professionali Turing)

## Funzionalità

- Benchmark matriciale con diverse dimensioni (1024x2048, 2048x2048, 4096x4096)
- Misurazione banda di memoria
- Comparazione con database GPU note
- Visualizzazione ASCII delle performance
- Analisi dettagliata e raccomandazioni

## Changelog

Vedere [CHANGELOG.md](CHANGELOG.md) per la storia completa delle modifiche.

## TODO

- [ ] Aggiungere supporto per test multi-GPU paralleli
- [ ] Implementare esportazione risultati in vari formati (JSON, CSV, PDF)
- [ ] Creare dashboard web per visualizzazione risultati storici
- [ ] Aggiungere supporto per benchmark personalizzati
- [ ] Implementare confronto automatico con benchmark precedenti
- [ ] Aggiungere profiling memoria dettagliato
- [ ] Sviluppare modalità batch per test automatizzati
- [ ] Integrare supporto per container Docker

## Contribuire

Le pull request sono benvenute. Per modifiche importanti, aprire prima una issue per discutere le modifiche proposte.

## Licenza

[MIT](https://choosealicense.com/licenses/mit/)
