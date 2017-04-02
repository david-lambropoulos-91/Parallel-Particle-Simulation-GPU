#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
#include <vector>
#include <string.h>

#define NUM_THREADS 256

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005
#define binsize (cutoff*4)


extern double size;
//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = dx * dx + dy * dy;
	if( r2 > cutoff*cutoff )
		return;
	//r2 = fmax( r2, min_r*min_r );
	r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
	double r = sqrt( r2 );

	//
	//  very simple short-range repulsive force
	//
	double coef = ( 1 - cutoff / r ) / r2 / mass;
	particle.ax += coef * dx;
	particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, int n, int* thread_offset, int* row_offset)
{
	// Get thread (particle) ID
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//if(tid >= n) return;
	
	int preparticles = row_offset[thread_offset[blockIdx.x]];
	int postparticles = row_offset[thread_offset[blockIdx.x +1]];

	int workingThreads = postparticles - preparticles;
	if(threadIdx.x >= workingThreads)
		return;

	//Total shared memory allocation: 6 kb
	__shared__ particle_t local[256];//2kb
	__shared__ particle_t haloAbove[256];
	__shared__ particle_t haloBelow[256];
	//The full halo allocation will generally not be used.


	int tHaloSize = 0;
	int bHaloSize = 0;
	int tHaloStart = 0;
	int bHaloStart = 0;
	int bHaloEnd = 0;

	local[threadIdx.x] = particles[preparticles + threadIdx.x]; 
	
	if(blockIdx.x > 0 )
	{
		tHaloStart = row_offset[thread_offset[blockIdx.x] - 1];
		tHaloSize = preparticles - tHaloStart;
		if(threadIdx.x < tHaloSize)
		//Create the upper halo region
			haloAbove[threadIdx.x] = particles[tHaloStart + threadIdx.x];
	}

	if(blockIdx.x < gridDim.x - 1 )
	{
		bHaloStart = row_offset[thread_offset[blockIdx.x + 1]];
		bHaloEnd = row_offset[thread_offset[blockIdx.x + 1] +1];
		bHaloSize = bHaloEnd - bHaloStart;
		if(threadIdx.x < bHaloSize)
		//Create the lower halo region
			haloBelow[threadIdx.x] = particles[bHaloStart + threadIdx.x];
	}

	__syncthreads();
	//Now do binning for all particles in every thread, 
	//and also bin the hal o regions if they exist.

	//NEW IDEA: Just gather the neighbors to the particle, no need to create 'bins'
	//Only check in halo area if you're in the right row
	//...Which really means just apply_force_gpu over the entire block.
	//Time complexity check: ... for thread blocks of size M (~256), 
	//We do M^2 checks to find neighbors
	//There are N/M +1 thread blocks, making the resulting complexity O(NM)~~ O(N) given M<256
	local[threadIdx.x].ax = local[threadIdx.x].ay = 0;

	for(int i = 0; i < workingThreads; i++){
		apply_force_gpu( local[threadIdx.x], local[i] );
	}
	for(int i = 0; i < tHaloSize; i++){
		apply_force_gpu(local[threadIdx.x], haloAbove[i]);
	}
	for(int i = 0; i < bHaloSize; i++){
		apply_force_gpu(local[threadIdx.x], haloBelow[i]);
	}
	//Potentially need to synchronize here...? Maybe not.
	//Copy shared local back into global here
	particles[preparticles + threadIdx.x] = local[threadIdx.x];
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

	// Get thread (particle) ID
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= n) return;

	particle_t * p = &particles[tid];
	//
	//  slightly simplified Velocity Verlet integration
	//  conserves energy better than explicit Euler method
	//
	p->vx += p->ax * dt;
	p->vy += p->ay * dt;
	p->x  += p->vx * dt;
	p->y  += p->vy * dt;

	//
	//  bounce from walls
	//
	while( p->x < 0 || p->x > size )
	{
		p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
		p->vx = -(p->vx);
	}
	while( p->y < 0 || p->y > size )
	{
		p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
		p->vy = -(p->vy);
	}

}

int main( int argc, char **argv )
{
	// This takes a few seconds to initialize the runtime
	cudaThreadSynchronize(); 

	if( find_option( argc, argv, "-h" ) >= 0 )
	{
		printf( "Options:\n" );
		printf( "-h to see this help\n" );
		printf( "-n <int> to set the number of particles\n" );
		printf( "-o <filename> to specify the output file name\n" );
		return 0;
	}
	
	int n = read_int( argc, argv, "-n", 1000 );

	char *savename = read_string( argc, argv, "-o", NULL );
	
	FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	particle_t *sorted = (particle_t*) malloc( n * sizeof(particle_t) );
	double copy_time_accum = 0;

	// GPU particle data structure
	particle_t * d_particles;
	cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

	double size = sqrt(density * n); //This is from common.cpp
	int num_bin_row = ceil(size / binsize);
	int blks = 0;

	set_size( n );

	int* row_sizes = (int*) malloc(sizeof(int) * num_bin_row);
	int* row_offsets = (int*) malloc(sizeof(int) * num_bin_row+1);
	int* row_capacity = (int*) malloc(sizeof(int) * num_bin_row);
	//This is most likely overallocation, but there is no way it will exceed this boundary.
	int* thread_rows = (int*) malloc(sizeof(int) * num_bin_row);
	int* thread_offset = (int*) malloc(sizeof(int) * num_bin_row+1);
	printf("Malloced arrays properly\n");
	std::vector<particle_t> bins[num_bin_row];

	init_particles( n, particles );

	//put particles in row bins according to their y position
	for (int j = 0; j < n; j++) {
		int y = floor(particles[j].y / binsize);
		bins[y].push_back(particles[j]);
	}
	printf("Bins sorted\n");
	//Create the sorted particle array and row sizes/offsets array.
	int num_rows = 0;
	int accum = 0;
	thread_rows[0] = 0;
	row_offsets[0] = bins[0].size();
	for(int i = 0; i < num_bin_row; i++){
		row_sizes[i] = bins[i].size();
		memcpy(&sorted[row_sizes[i]], bins[i].data(), sizeof(particle_t) * bins[i].size());
		row_offsets[i+1] = row_offsets[i] + bins[i].size();
		//Latter part of loop handles cuda thread row allocation
		accum += row_sizes[i];
		if(accum > 256){
			if(num_rows == 0) {
				printf("Num_rows = 0\n");
				return -1;
			}
			thread_rows[blks] = num_rows;
			blks++;
			thread_offset[blks] = i;
			accum = row_sizes[i];
		}
		num_rows++;
	}
	printf("sorted array populated\n");
	//We can only malloc space for device thread offsets after we know how large blks is.
	int* d_toff;
	int* d_roff;
	cudaMalloc((void **) &d_toff, blks * sizeof(int));
	cudaMalloc((void **) &d_roff, (num_bin_row+1) * sizeof(int));

	//
	//  simulate a number of time steps
	//
	cudaThreadSynchronize();
	double simulation_time = read_timer( );

	for( int step = 0; step < NSTEPS; step++ )
	{
		cudaThreadSynchronize();
		double copy_time = read_timer( );

		// Copy the *Sorted* particles to the GPU
		cudaMemcpy(d_particles, sorted, n * sizeof(particle_t), cudaMemcpyHostToDevice);
		// We also want thread offsets and row offsets
		cudaMemcpy(d_toff, thread_offset, blks * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_roff, row_offsets, (num_bin_row+1) * sizeof(int), cudaMemcpyHostToDevice);
		printf("particles and offsets memcpyied\n");
		cudaThreadSynchronize();
		copy_time = read_timer( ) - copy_time;
		copy_time_accum+= copy_time;
		compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n, d_toff, d_roff);
		printf("Compute_Forces complete\n");
		move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
		printf("move_gpu complete\n");
		//This may cause a lot of overhead... 
		//If we manage to do bookkeeping within move_gpu, we will not need this copy overhead..
		cudaMemcpy(sorted, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
		
		//
		//  save if necessary
		//
		if( fsave && (step%SAVEFREQ) == 0 ) {
		// Copy the particles back to the CPU
			cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
			save( fsave, n, particles);
		}
		for (int i = 0; i < num_bin_row; i++)
			bins[i].clear();
		//put particles in row bins according to their y position
		for (int j = 0; j < n; j++) {
			int y = floor(sorted[j].y / binsize);
			bins[y].push_back(sorted[j]);
		}
		//Create the sorted particle array and row sizes/offsets array.
		int num_rows = 0;
		thread_rows[0] = 0;
		row_offsets[0] = bins[0].size();
		for(int i = 0; i < num_bin_row; i++){
			row_sizes[i] = bins[i].size();
			memcpy(&sorted[row_sizes[i]], bins[i].data(), sizeof(particle_t) * bins[i].size());
			row_offsets[i+1] = row_offsets[i] + bins[i].size();
			//Latter part of loop handles cuda thread row allocation
			accum += row_sizes[i];
			if(accum > 256){
				if(num_rows == 0) {
					printf("Num_rows = 0\n");
					return -1;
				}
				thread_rows[blks] = num_rows;
				blks++;
				thread_offset[blks] = i;
				accum = row_sizes[i];
			}
			num_rows++;
		}

	}
	cudaThreadSynchronize();
	simulation_time = read_timer( ) - simulation_time;
	
	printf( "CPU-GPU copy time = %g seconds\n", copy_time_accum);
	printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
	
	free( particles );
	free( sorted );
	free( row_offsets );
	free( row_sizes );
	free( row_capacity );
	free( thread_rows );
	free( thread_offset );
	cudaFree(d_roff);
	cudaFree(d_toff);
	cudaFree(d_particles);
	if( fsave )
		fclose( fsave );
	
	return 0;
}
