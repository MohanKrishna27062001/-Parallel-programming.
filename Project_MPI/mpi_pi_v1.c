#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int i, rank, size, NUMSTEPS;
    double x, pi, sum = 0.0, local_sum = 0.0, step;
    struct timespec start, end;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    // Read NUMSTEPS from the command-line argument
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <NUMSTEPS>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }
    NUMSTEPS = atoi(argv[1]);

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start);  // Start timing (for master)
    }

    step = 1.0 / (double) NUMSTEPS;

    // Each process calculates a portion of the sum
    for (i = rank; i < NUMSTEPS; i += size) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Collect the local sums from all processes and sum them at the master process
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Final calculation and timing on the master process
    if (rank == 0) {
        pi = step * sum;
        clock_gettime(CLOCK_MONOTONIC, &end);  // End timing

        u_int64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

        printf("Calculated PI = %.20f with NUMSTEPS = %d\n", pi, NUMSTEPS);
        printf("Elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
