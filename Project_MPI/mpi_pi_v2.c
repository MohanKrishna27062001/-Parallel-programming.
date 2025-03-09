#include <stdio.h>
#include <stdlib.h>
#include<stdint.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int i, rank, size, chunk_size, start_idx, end_idx, NUMSTEPS;
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

    // Split work evenly across processes
    chunk_size = NUMSTEPS / size;
    start_idx = rank * chunk_size;
    end_idx = (rank == size - 1) ? NUMSTEPS : start_idx + chunk_size;  // Last process may get extra steps

    // Each process calculates a chunk of the sum
    for (i = start_idx; i < end_idx; i++) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    if (rank == 0) {
        sum = local_sum;  // Initialize sum with the local_sum of the root process

        // Receive local sums from each other process and accumulate
        for (int i = 1; i < size; i++) {
            double temp_sum;
            MPI_Recv(&temp_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += temp_sum;
        }

        // Final calculation and timing on the master process
        pi = step * sum;
        clock_gettime(CLOCK_MONOTONIC, &end);  // End timing

        uint64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);

        printf("Calculated PI = %.20f with NUMSTEPS = %d\n", pi, NUMSTEPS);
        printf("Elapsed time = %llu nanoseconds\n", (long long unsigned int)diff);

    } else {
        // Other processes send their local sums to the root process
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
