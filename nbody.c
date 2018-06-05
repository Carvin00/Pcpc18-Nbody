#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define SOFTENING 1e-9f
#define MAX_PROC 200

typedef struct{ 	//The following struct has the information for every single body 
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;
} Body;

MPI_Datatype BodyMPI; //New MPI type needed to use the body struct during MPI message exchange 

//function prototypes 
void getArgs(int argc, char **argv);	//Get the command line arguments 
void randomize(); 			//Inizializes the bodies with random values
void bodyForce();			//Calculate the bodies velocity
void updatePos();			//Update the bodies posigtions
void falseGather();			//Simulate AllGather when it is not possible to use collective methods
void printBodies(double time); 		//Print the information about the bodies at the current iteration

//MPI variables
int worldRank; 		//Processor rank in MPI_COMM_WORLD
int worldSize;		//Processor number in MPI_COMM_WORLD
MPI_Status status;

//Command line arguments
int totBodies;		//Number of bodies used during the simulation
int nIters;		//Number of iterations used for the simulation
int test=0;		//If 1 status of bodies will be printed
int outputFreq=1;	//Frequency of output printed to check the simulation progress only if test=1

//Simulation Variables
int nBodies;		//Number of bodies for each processor 
int bottomBody;		//First body for each process
int topBody;		//Last body for each process
int chunk;		//totBodies/worldSize used to calculate nBodies or procBodies
int reminder;		//totBodies%worldSize used to calculate nBodies or procBodies
float dt;		//Given time for each iteration 

Body *bodies; 		//Set of bodies

int main(int argc, char** argv) {
	
	double time;		//Current time
	double timeStart;	//Execution beginning time 
	double timeEnd;		//Execution ending time 

	//Inizialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);	//Each processor gets its rank
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);	//Number of processor in the current simulation

	//Command line arguments gathering 
	getArgs(argc, argv);

	//Bodies assignment for each processor
	reminder= totBodies%worldSize;
	chunk= totBodies/worldSize;
	if (reminder== 0) {
		nBodies=chunk;
		bottomBody= worldRank*chunk;
		topBody=((worldRank+1)*chunk)-1;
	} else if (reminder > 0) {
		if(worldRank< reminder) {
			nBodies= chunk+1;
			bottomBody= worldRank*chunk;
			topBody= ((worldRank+1)*chunk);
		} else if(worldRank>= reminder) {
			nBodies= chunk;
			bottomBody= worldRank*chunk+reminder;
			topBody= ((worldRank+1)*chunk)+reminder-1;
		}
	}
	if(test== 1) { 
		printf("Processor %d --- Assigned %d bodies from %d to %d\n", worldRank, nBodies, bottomBody, topBody);
	}	
	//Allocate necessary memory
	bodies= malloc(totBodies*sizeof(Body));

	//MPI commit of the new datatype
	MPI_Type_contiguous(6, MPI_FLOAT, &BodyMPI);
	MPI_Type_commit(&BodyMPI);

	randomize(); 	//Inizialize position and velocity of every body
	
	timeStart= MPI_Wtime();  //Start evaluation time
	
	if( worldRank== 0 && test== 1){
		printBodies(0.0);
	}

	int i;
	for(i= 1; i<= nIters; i++){
		time = i*dt;
		bodyForce();		//Update bodies velocity
		updatePos(); 		//Update bodies positions
		MPI_Barrier(MPI_COMM_WORLD);
		if(reminder== 0) { 	//Each processor get the next iteration bodies
			MPI_Allgather(bodies+ bottomBody, nBodies, BodyMPI, bodies, nBodies, BodyMPI, MPI_COMM_WORLD);
		} else {
			falseGather();
		} 
		if((i%outputFreq)== 0 && worldRank== 0 && test== 1) {
			printBodies(time);
		}
	}

	timeEnd= MPI_Wtime();
	if(worldRank== 0) {
		printf("Game Over -- Elapsed time: %f seconds \n", timeEnd-timeStart);
	}

	//MPI free memory
	MPI_Type_free(&BodyMPI);
	//Malloc free
	free(bodies);

	//MPI end
	MPI_Finalize();

	return 0;

}

void getArgs(int argc, char **argv) {
	
	if( argc == 5) {
		totBodies= strtol(argv[1], NULL, 10);
		nIters= strtol(argv[2], NULL, 10);
		test= atoi(argv[3]);
		outputFreq= strtol(argv[3], NULL, 10);
		dt=0.1f;				//dt isn't a line argument, but it can easily become one to customize the simulation ( argc == 6 and dt=strtol(argv[4], NULL, 10);)
	} else if (argc==3) {
		totBodies= strtol(argv[1], NULL, 10);
		nIters= strtol(argv[2], NULL, 10);
		dt=0.1f;
	} else	{
		if(worldRank== 0) {
			printf("You must insert 2 arguments and 2 optional: \n");
			printf("1-- Number of bodies;\n");
			printf("2-- Number of iterations;\n");
			printf("3-- OPTIONAL 1 if it's a test, status of processor and bodies will be printed\n");
			printf("4-- OPTIONAL Frequency of update about the simulation. Only with test=1\n");
		}
	}

	if( totBodies<= 0 || nIters<= 0) { //Checking the correctness of the arguments
		MPI_Finalize();
		exit(0);
	}
}

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

void printBodies(double time) {
	int part;
	printf("Current time: %.2f\n", time);

	for(part= 0; part< totBodies; part++){
		printf("Body:%3d --- X:%10.3e -- Y:%10.3e -- Z:%10.3e -- Vx:%10.3e -- Vy:%10.3e -- Vz:%10.3e\n", part, bodies[part].x,  bodies[part].y, bodies[part].z, bodies[part].vx, bodies[part].vy, bodies[part].vz);
	}
}

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
