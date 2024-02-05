# Rendering: Parallel Programming for Machine Learning
Il progetto si focalizza sulla verifica di specifiche funzioni implementate in C++ e CUDA 
per la parallelizzazione del metodo di compositing.


<img src="results/img/seq/10000.png" alt="Risultato del test con 10000 piani" width="256" height="256" style="display:inline-block; margin-right: 10px;">
<img src="results/img/par/10000.png" alt="Risultato del test con 10000 piani" width="256" height="256" style="display:inline-block; margin-right: 10px;">
<img src="results/img/cuda/10000.png" alt="Risultato del test con 10000 piani" width="256" height="256" style="display:inline-block; margin-right: 10px;">
<img src="results/img/cuda_color/10000.png" alt="Risultato del test con 10000 piani" width="256" height="256" style="display:inline-block; margin-right: 10px;">

## SRC
Questa directory contiene le principali classi utilizzate per l'esecuzione dei test:

* test.h - File utilizzato per la modifica dei valori di test.
* renderer.cu - File principale per lo sviluppo dei metodi di parallelizzazione.

## RESULT
Questa cartella raccoglie i risultati ottenuti dai vari test effettuati.

## OUT  
Cartella contente la relazione per il documento finale.
