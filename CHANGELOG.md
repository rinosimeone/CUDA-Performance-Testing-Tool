# Changelog

Tutte le modifiche notevoli al progetto saranno documentate in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e questo progetto aderisce al [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-02-17

### Aggiunto
- Visualizzazione grafica dei risultati con matplotlib
- Benchmark comparativo CPU vs GPU
- Calcolo e visualizzazione dello speedup
- Supporto per diverse dimensioni di matrici nei test
- Grafici di performance salvati in PNG
- Dipendenze matplotlib e numpy

### Modificato
- Aggiornato README con nuove funzionalità
- Migliorata la struttura dei benchmark
- Ottimizzato il codice per test comparativi
- Aggiornati i requisiti del progetto

### Caratteristiche Tecniche
- Test con matrici 1000x1000 (100 iterazioni)
- Test con matrici 2000x2000 (50 iterazioni)
- Test con matrici 4000x4000 (25 iterazioni)
- Visualizzazione grafica tripla (confronto, speedup, barre)
- Sincronizzazione CUDA per timing accurati

## [1.0.0] - 2025-02-09

### Aggiunto
- Implementazione iniziale del benchmark CUDA
- Database GPU di riferimento con performance note
- Visualizzazione ASCII dei risultati
- Analisi dettagliata delle performance
- Comparazione con altre GPU
- Analisi dello scaling delle performance
- Raccomandazioni specifiche per il carico di lavoro
- Supporto per CUDA 12.x
- README.md con documentazione completa

### Modificato
- Migliorata la precisione del benchmark con warm-up
- Ottimizzata la gestione della memoria durante i test
- Raffinata l'analisi comparativa con GPU simili

### Caratteristiche Tecniche
- Supporto per matrici fino a 4096x4096
- Misurazione banda memoria
- Analisi efficienza scaling
- Visualizzazione ASCII performance
- Raccomandazioni ottimizzazione
