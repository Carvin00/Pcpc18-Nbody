# N-body Simulation

***

## Programmazione Concorrente, Parallela e su Cloud
### Università degli Studi di Salerno
#### *Anno Accademico 2017/2018*

**Professore:** _Vittorio Scarano_
**Dottore:** _Carmine Spagnuolo_
**Studente:** _Carmine Vincenzo Russo_


---

## Problem Statement

In un problema n-body, abbiamo bisogno di trovare la posizione e la velocità di una collezione di corpi/particelle che interaggiscono in un periodo ti tempo.
Per esempio, un astrofisico potrebbe voler conoscere la posizione a la velocità di una collezione di stelle, mentre un chimico potrebbe voler conoscere la posizione e la velocità di una collezione di molecole o atomi.
Una soluzione n-body è prodotta da un programma che trova le soluzioni ad un prolema n-body simulando il comportamento dei corpi/particelle.
L'input del programma è il numero di corpi e il numero di iterazioni su cui effettuare le iterazioni. Sarà compito del programma generare in maniera pseudocasuale la posizione e la velocità di tutti i corpi richiesti dal numero di input in un ambiente tridimensinale. L'output del programma sarà il tempo totale utilizzato dal programma per eseguire la simulazione del problema n-body. 
Sono inoltre previsti due argomenti di input aggiuntivi: un flag di test e la frequenza di stampa, se il flag test è settato ad 1 verranno effettuate delle stempe di controllo per la divisione dei corpi sui processori e l'aggiornamento dei corpi all'interno dello spazio tridimensionale. Le stampe dei corpi verranno effettuate ogni X iterazioni della simulazione, dove X è la frequenza di stampa.

## Soluzione proposta

La soluzione proposta considera solo l'approccio n^2 rispetto al numero di corpi/particelle scelti. 
Sono state utilizzate funzioni __Scatter__ e __AllGather__ con l'aggiunta delle funzioni __send__ e __recv__ di **MPI** per realizzare un tipo di comunicazione Collective bloccante. 
I test sono stati effettuati sulle istanze di AWS **m4.xlarge**.

### Implementazione

Lo scopo del lavoro svolto è stato quello di parallelizzare il problema proposto partizionandolo in maniera equa su tutti i processori disponibili nel sistema.
Di seguito è descritto l'approccio utilizzato al fine di ottenere una distribuzione equa

```c
reminder= totBodies%worldSize;
chunk= totBodies/worldRank;
if (reminder== 0) {
	nBodies= chunk;
	bottomBody= worldRank* chunk;
	topBody= ((worldRank+1)* chunk)- 1;
} else if(reminder > 0 ) {
	if (worldRank < reminder) {
		nBodies= chunk+1;
		bottomBody= worldRank* chunk;
		topBody= ((worldRank+1)* chunk);
	} else if{
		nBodies= chunk;
		bottomBody= worldRank* chunk+ reminder;
		topBody= ((worldRank+1)* chunk)+ reminder- 1;
		}
}
```

Questo approccio ci permette di distribuire equamente il carico su tutti i processori evitando che ci siano processori più carichi di altri. Infatti assegiamo a tutti i processori lo stesso numero di corpi e se abbiamo corpi rimanenti (reminder) dopo questa assegnazione, questi li distribuiamo 1 a processore fino a terminarli seguendo i rank assegnati ai processori stessi. 
