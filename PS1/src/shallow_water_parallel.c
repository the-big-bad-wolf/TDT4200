#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size = 10.0,
    gravity = 9.81,
    density = 997.0;

real_t
    *mass[2] = {NULL, NULL},
    *mass_velocity_x[2] = {NULL, NULL},
    *velocity_x = NULL,
    *acceleration_x = NULL,
    dx,
    dt;

int rank, worldsize;

#define PN(x) mass[0][(x)]
#define PN_next(x) mass[1][(x)]
#define PNU(x) mass_velocity_x[0][(x)]
#define PNU_next(x) mass_velocity_x[1][(x)]
#define U(x) velocity_x[(x)]
#define DU(x) acceleration_x[(x)]

void time_step(void);
void boundary_condition(real_t *domain_variable, int sign);
void domain_init(void);
void domain_save(int_t iteration);
void domain_finalize(void);
void communicate_border(void);

void swap(real_t **m1, real_t **m2)
{
    real_t *tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

int main(int argc, char **argv)
{
    // TODO 1 Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    // TODO 2 Parse arguments in the rank 0 processes
    // and broadcast to other processes

    int items = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT64_T, MPI_INT64_T, MPI_INT64_T};
    MPI_Datatype mpi_options_type;

    MPI_Aint offsets[3];
    offsets[0] = offsetof(OPTIONS, N);
    offsets[1] = offsetof(OPTIONS, max_iteration);
    offsets[2] = offsetof(OPTIONS, snapshot_frequency);

    MPI_Type_create_struct(items, blocklengths, offsets, types, &mpi_options_type);

    MPI_Type_commit(&mpi_options_type);

    OPTIONS options;

    if (rank == 0)
    {
        options = *parse_args(argc, argv);
        if (!&options)
        {
            fprintf(stderr, "Argument parsing failed\n");
            exit(1);
        }
    }

    MPI_Bcast(&options, 1, mpi_options_type, 0, MPI_COMM_WORLD);

    N = options.N;
    max_iteration = options.max_iteration;
    snapshot_frequency = options.snapshot_frequency;

    MPI_Type_free(&mpi_options_type);

    // TODO 3 Allocate space for each process' sub-grid
    // and initialize data for the sub-grid
    domain_init();

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // TODO 7 Communicate border values
        communicate_border();

        // TODO 5 Boundary conditions
        boundary_condition(mass[0], 1);
        boundary_condition(mass_velocity_x[0], -1);

        // TODO 4 Time step calculations
        time_step();

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            // TODO 6 MPI I/O
            domain_save(iteration);
        }

        swap(&mass[0], &mass[1]);
        swap(&mass_velocity_x[0], &mass_velocity_x[1]);
    }
    domain_finalize();

    // TODO 1 Finalize MPI
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}

void time_step(void)
{
    // TODO 4 Time step calculations
    int_t workload = N / worldsize;
    for (int_t x = 0; x <= workload + 1; x++)
    {
        DU(x) = PN(x) * U(x) * U(x) + 0.5 * gravity * PN(x) * PN(x) / density;
    }

    for (int_t x = 1; x <= workload; x++)
    {
        PNU_next(x) = 0.5 * (PNU(x + 1) + PNU(x - 1)) - dt * ((DU(x + 1) - DU(x - 1)) / (2 * dx));
    }

    for (int_t x = 1; x <= workload; x++)
    {
        PN_next(x) = 0.5 * (PN(x + 1) + PN(x - 1)) - dt * ((PNU(x + 1) - PNU(x - 1)) / (2 * dx));
    }

    for (int_t x = 1; x <= workload; x++)
    {
        U(x) = PNU_next(x) / PN_next(x);
    }
}

void boundary_condition(real_t *domain_variable, int sign)
{
// TODO 5 Boundary conditions
#define VAR(x) domain_variable[(x)]
    if (rank == 0)
    {
        VAR(0) = sign * VAR(2);
    }
    if (rank == worldsize - 1)
    {
        VAR(N / worldsize + 1) = sign * VAR(N / worldsize - 1);
    }
#undef VAR
}

void domain_init(void)
{
    // TODO 3 Allocate space for each process' sub-grid
    // and initialize data for the sub-grid
    int_t workload = N / worldsize;

    mass[0] = calloc(workload + 2, sizeof(real_t));
    mass[1] = calloc(workload + 2, sizeof(real_t));

    mass_velocity_x[0] = calloc(workload + 2, sizeof(real_t));
    mass_velocity_x[1] = calloc(workload + 2, sizeof(real_t));

    velocity_x = calloc(workload + 2, sizeof(real_t));
    acceleration_x = calloc(workload + 2, sizeof(real_t));

    // Data initialization
    for (int_t x = 1; x <= workload; x++)
    {
        PN(x) = 1e-3;
        PNU(x) = 0.0;

        real_t c = x + rank * workload - N / 2;
        if (sqrt(c * c) < N / 20.0)
        {
            PN(x) -= 5e-4 * exp(
                                -4 * pow(c, 2.0) / (real_t)(N));
        }

        PN(x) *= density;
    }

    dx = domain_size / (real_t)N;
    dt = 0.1 * dx;
}

void domain_save(int_t iteration)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    // TODO 6 MPI I/O

    MPI_File out;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &out);

    MPI_Offset offset = rank * sizeof(real_t) * (N / worldsize);
    MPI_File_write_at_all(out, offset, &mass[0][1], (N / worldsize), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&out);
}

void domain_finalize(void)
{
    free(mass[0]);
    free(mass[1]);
    free(mass_velocity_x[0]);
    free(mass_velocity_x[1]);
    free(velocity_x);
    free(acceleration_x);
}

void communicate_border()
{
    if (worldsize == 1)
    {
        return;
    }
    int rightrank;
    int leftrank;
    if (rank == worldsize - 1)
    {
        rightrank = 0;
    }
    else
    {
        rightrank = rank + 1;
    }
    if (rank == 0)
    {
        leftrank = worldsize - 1;
    }
    else
    {
        leftrank = rank - 1;
    }
    MPI_Send(&mass[0][N / worldsize], 1, MPI_DOUBLE, rightrank, 0, MPI_COMM_WORLD);
    MPI_Send(&mass_velocity_x[0][N / worldsize], 1, MPI_DOUBLE, rightrank, 1, MPI_COMM_WORLD);
    MPI_Recv(&mass[0][0], 1, MPI_DOUBLE, leftrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&mass_velocity_x[0][0], 1, MPI_DOUBLE, leftrank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Send(&mass[0][1], 1, MPI_DOUBLE, leftrank, 0, MPI_COMM_WORLD);
    MPI_Send(&mass_velocity_x[0][1], 1, MPI_DOUBLE, leftrank, 1, MPI_COMM_WORLD);
    MPI_Recv(&mass[0][N / worldsize + 1], 1, MPI_DOUBLE, rightrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&mass_velocity_x[0][N / worldsize + 1], 1, MPI_DOUBLE, rightrank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}