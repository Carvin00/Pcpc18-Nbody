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

Questo approccio ci permette di distribuire equamente il carico su tutti i processori evitando che ci siano processori più carichi di altri. Infatti assegiamo a tutti i processori lo stesso numero di corpi e se abbiamo corpi rimanenti (reminder) dopo questa assegnazione, questi li distribuiamo assegnando 1 corpo per processore fino a terminarli seguendo i rank assegnati ai processori stessi. 

I valori iniziali delle posizioni e delle velocità dei corpi vengono generati in maniera pseudocasuale. Il processo di generazione è suddiviso tra tutti i processori seguendo le partizioni prima calcolate dal bottomBody al topBody.

```c
void randomize() {
	int i;
	for(i = bottomBody; i<= topBody; i++) {
		bodies[i].x = 2.0f * (rand()/ (float)RAND_MAX) - 1.0f;
		bodies[i].y = 2.0f * (rand()/ (float)RAND_MAX) - 1.0f;
		bodies[i].z = 2.0f * (rand()/ (float)RAND_MAX) - 1.0f;
		bodies[i].vx = 2.0f * (rand()/ (float)RAND_MAX) - 1.0f;
		bodies[i].vy = 2.0f * (rand()/ (float)RAND_MAX) - 1.0f;
		bodies[i].vz = 2.0f * (rand()/ (float)RAND_MAX) - 1.0f;
	}
	if (reminder== 0) {
		MPI_Allgather(bodies+ bottomBody, nBodies, BodyMPI, bodies, nBodies, BodyMPI, MPI_COMM_WORLD);
	} else {
		falseGather();
	}

}
```

Conclusa l'iniziazializzazione dei corpi in ogni processore è necessario fare la propagazione dei corpi generati con tutti i processori che stanno partecipando alla risoluzione del problema. Nel caso di reminder uguale a zero è posibile utilizzare la funzione collettiva **AllGather** altrimenti utilizziamo la funzione _falseGather_ che è una simulazione dell'AllGather realizzata utilizzando **send** e **revc**.

Bisogna quindi calcolare le nuove posizioni e velocità dei corpi e successivamente aggiornare le posizioni. E' sempre necessario poi propagare tutte le nuove informazioni calcolate tramite **AllGather** o _falseGather_.

```c
void bodyForce() {
	int i, j;
	for(i= bottomBody; i<= topBody; i++) {
		float fx= 0.0f;
		float fy= 0.0f;
		float fz= 0.0f;
		for(j= 0; j< totBodies; j++) {
			float dx = bodies[j].x- bodies[i].x;
			float dy = bodies[j].y- bodies[i].y;
			float dz = bodies[j].z- bodies[i].z;
			float distSqr= dx* dx+ dy* dy+ dz* dz+ SOFTENING;
			float invDist= 1.0f/sqrtf(distSqr);
			float invDist3= invDist* invDist* invDist;

			fx+= dx* invDist3;
			fy+= dy* invDist3;
			fz+= dz* invDist3;
		}
		bodies[i].vx+= dt*fx;
		bodies[i].vy+= dt*fy;
		bodies[i].vz+= dt*fz;
	}
}

void updatePos() {
	int i; 
	for(i= bottomBody; i<= topBody; i++) {
		bodies[i].x+= bodies[i].vx*dt;
		bodies[i].y+= bodies[i].vy*dt;
		bodies[i].z+= bodies[i].vz*dt;
	}
}
```

La comunicazione che non sfrutta è le funzioni collettive è stato realizzato come segue:

```c
void falseGather() {
	int i;
	for(i= 0; i< worldSize; i++) {
		if(worldRank== i) {
			int j;
			for(j= 0; j< worldSize; j++) {
				if(j!= worldRank) {
					MPI_Send(bodies+ bottomBody, nBodies, BodyMPI, j, 1, MPI_COMM_WORLD);
					
				}
			}
		} else {
			int rec, start;
			if(i< reminder) {
				rec= chunk+1;
				start= rec*i; 
			} else {
				rec= chunk;
				start= rec* i+ reminder;
			}
			MPI_Recv(bodies+ start, rec, BodyMPI, i, 1, MPI_COMM_WORLD, &status);

		}
	}
}
```

L'implementazione sdd:
