// ---------------------------------------------------------
// TDT4200 Parallel Computing - Graded CUDA
// ---------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <cooperative_groups.h>

#include "../inc/argument_utils.h"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

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
    *h_mass_0 = NULL,
    *h_mass_1 = NULL,
    *d_mass_0 = NULL,
    *d_mass_1 = NULL,

    *h_mass_velocity_x_0 = NULL,
    *h_mass_velocity_x_1 = NULL,
    *d_mass_velocity_x_0 = NULL,
    *d_mass_velocity_x_1 = NULL,

    *h_mass_velocity_y_0 = NULL,
    *h_mass_velocity_y_1 = NULL,
    *d_mass_velocity_y_0 = NULL,
    *d_mass_velocity_y_1 = NULL,

    *h_mass_velocity = NULL,
    *d_mass_velocity = NULL,

    *h_velocity_x = NULL,
    *d_velocity_x = NULL,
    *h_velocity_y = NULL,
    *d_velocity_y = NULL,

    *h_acceleration_x = NULL,
    *d_acceleration_x = NULL,
    *h_acceleration_y = NULL,
    *d_acceleration_y = NULL,
    dx,
    dt;

size_t array_size; // Size of the arrays used to store the data points

#define PN(y, x) mass_0[(y) * (N + 2) + (x)]
#define PN_next(y, x) mass_1[(y) * (N + 2) + (x)]
#define PNU(y, x) mass_velocity_x_0[(y) * (N + 2) + (x)]
#define PNU_next(y, x) mass_velocity_x_1[(y) * (N + 2) + (x)]
#define PNV(y, x) mass_velocity_y_0[(y) * (N + 2) + (x)]
#define PNV_next(y, x) mass_velocity_y_1[(y) * (N + 2) + (x)]
#define PNUV(y, x) mass_velocity[(y) * (N + 2) + (x)]
#define U(y, x) velocity_x[(y) * (N + 2) + (x)]
#define V(y, x) velocity_y[(y) * (N + 2) + (x)]
#define DU(y, x) acceleration_x[(y) * (N + 2) + (x)]
#define DV(y, x) acceleration_y[(y) * (N + 2) + (x)]

#define cudaErrorCheck(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void time_step_1(real_t *velocity_x, real_t *velocity_y,
                            real_t *acceleration_x, real_t *acceleration_y,
                            real_t *mass_velocity_x_0, real_t *mass_velocity_y_0,
                            real_t *mass_velocity, real_t *mass_0, int_t N);

__global__ void time_step_2(real_t *acceleration_x, real_t *acceleration_y,
                            real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
                            real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
                            real_t *mass_velocity, real_t *mass_0, real_t *mass_1, real_t dx, real_t dt, int_t N);

__global__ void time_step(real_t *velocity_x, real_t *velocity_y,
                          real_t *acceleration_x, real_t *acceleration_y,
                          real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
                          real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
                          real_t *mass_velocity, real_t *mass_0, real_t *mass_1, real_t dx, real_t dt, int_t N);

// TODO: Rewrite boundary_condition as a device function.
__device__ void boundary_condition(real_t *domain_variable, int sign, int_t N, int x_index, int y_index);
void domain_init(void);
void domain_save(int_t iteration);
void domain_finalize(void);

// Pthreads threaded domain save function
void *domain_save_threaded(void *iter);

void swap(real_t **t1, real_t **t2)
{
    real_t *tmp;
    tmp = *t1;
    *t1 = *t2;
    *t2 = tmp;
}

int main(int argc, char **argv)
{

    OPTIONS *options = parse_args(argc, argv);
    if (!options)
    {
        fprintf(stderr, "Argument parsing failed\n");
        exit(1);
    }

    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    // Standard block and grid layout
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    uint threadsPerBlockDim = floor(sqrt(deviceProp.maxThreadsPerBlock)); // Max 1024 threads per block --> sqrt(1024)=32 threads per block dimension
    dim3 blockLayout = {threadsPerBlockDim, threadsPerBlockDim};

    uint blocksPerDim = ceil((N + 2.0) / threadsPerBlockDim); // Enough blocks to cover the whole grid and makes sure number of blocks is rounded up
    dim3 gridLayout = {blocksPerDim, blocksPerDim};
    // end

    // CUDA cooperative groups block and grid layout
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)time_step);

    uint coop_threadsPerBlockDim = floor(sqrt(blockSize));
    dim3 coop_blockLayout = {coop_threadsPerBlockDim, coop_threadsPerBlockDim};

    uint coop_blocksPerDim = ceil((N + 2.0) / coop_threadsPerBlockDim);
    dim3 coop_gridLayout = {coop_blocksPerDim, coop_blocksPerDim};
    // end

    void *
        kernel_args[14] = {&d_velocity_x,
                           &d_velocity_y,
                           &d_acceleration_x,
                           &d_acceleration_y,
                           &d_mass_velocity_x_0,
                           &d_mass_velocity_x_1,
                           &d_mass_velocity_y_0,
                           &d_mass_velocity_y_1,
                           &d_mass_velocity,
                           &d_mass_0,
                           &d_mass_1,
                           &dx,
                           &dt,
                           &N};

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {

        // TODO: Launch time_step kernels

        // Standard version
        /*
        time_step_1<<<gridLayout, blockLayout>>>(d_velocity_x, d_velocity_y,
                                                 d_acceleration_x, d_acceleration_y,
                                                 d_mass_velocity_x_0,
                                                 d_mass_velocity_y_0,
                                                 d_mass_velocity, d_mass_0, N);

        time_step_2<<<gridLayout, blockLayout>>>(
            d_acceleration_x, d_acceleration_y,
            d_mass_velocity_x_0, d_mass_velocity_x_1,
            d_mass_velocity_y_0, d_mass_velocity_y_1,
            d_mass_velocity, d_mass_0, d_mass_1, dx, dt, N);
        */

        // Cooperative groups version
        cudaErrorCheck(cudaLaunchCooperativeKernel((void *)time_step, coop_gridLayout, coop_blockLayout, kernel_args))

            if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld, (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            // TODO: Copy the masses from the device to host prior to domain_save
            cudaMemcpy(h_mass_0, d_mass_0, array_size, cudaMemcpyDeviceToHost);
            domain_save(iteration);
        }

        // TODO: Swap device buffer pointers between iterations
        swap(&d_mass_0, &d_mass_1);
        swap(&d_mass_velocity_x_0, &d_mass_velocity_x_1);
        swap(&d_mass_velocity_y_0, &d_mass_velocity_y_1);
    }

    domain_finalize();

    exit(EXIT_SUCCESS);
}

// TODO: Rewrite this function as one or more CUDA kernels
// ---------------------------------------------------------
// To ensure correct results, the participating threads in the thread
// grid must be synchronized after calculating the accelerations (DU, DV).
// If the grid is not synchronized, data dependencies cannot be guaranteed.

__global__ void time_step_1(real_t *velocity_x, real_t *velocity_y,
                            real_t *acceleration_x, real_t *acceleration_y,
                            real_t *mass_velocity_x_0,
                            real_t *mass_velocity_y_0,
                            real_t *mass_velocity, real_t *mass_0, int_t N)
{
    int x_index = threadIdx.x + blockIdx.x * blockDim.x;
    int y_index = threadIdx.y + blockIdx.y * blockDim.y;

    boundary_condition(mass_0, 1, N, x_index, y_index);
    boundary_condition(mass_velocity_x_0, -1, N, x_index, y_index);
    boundary_condition(mass_velocity_y_0, -1, N, x_index, y_index);

    if ((1 <= x_index) && (x_index <= N) && (1 <= y_index) && (y_index <= N))
    {
        U(y_index, x_index) = PNU(y_index, x_index) / PN(y_index, x_index);
        V(y_index, x_index) = PNV(y_index, x_index) / PN(y_index, x_index);
        PNUV(y_index, x_index) = PN(y_index, x_index) * U(y_index, x_index) * V(y_index, x_index);
    }

    if ((x_index <= N + 1) && (y_index <= N + 1))
    {
        DU(y_index, x_index) = PN(y_index, x_index) * U(y_index, x_index) * U(y_index, x_index) + 0.5 * gravity * (PN(y_index, x_index) * PN(y_index, x_index) / density);
        DV(y_index, x_index) = PN(y_index, x_index) * V(y_index, x_index) * V(y_index, x_index) + 0.5 * gravity * (PN(y_index, x_index) * PN(y_index, x_index) / density);
    }
}

__global__ void time_step_2(
    real_t *acceleration_x, real_t *acceleration_y,
    real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
    real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
    real_t *mass_velocity, real_t *mass_0, real_t *mass_1, real_t dx, real_t dt, int_t N)
{
    int x_index = threadIdx.x + blockIdx.x * blockDim.x;
    int y_index = threadIdx.y + blockIdx.y * blockDim.y;

    if ((1 <= x_index) && (x_index <= N) && (1 <= y_index) && (y_index <= N))
    {
        PNU_next(y_index, x_index) = 0.5 * (PNU(y_index, x_index + 1) + PNU(y_index, x_index - 1)) - dt * ((DU(y_index, x_index + 1) - DU(y_index, x_index - 1)) / (2 * dx) + (PNUV(y_index, x_index + 1) - PNUV(y_index, x_index - 1)) / (2 * dx));
        PNV_next(y_index, x_index) = 0.5 * (PNV(y_index + 1, x_index) + PNV(y_index - 1, x_index)) - dt * ((DV(y_index + 1, x_index) - DV(y_index - 1, x_index)) / (2 * dx) + (PNUV(y_index + 1, x_index) - PNUV(y_index - 1, x_index)) / (2 * dx));
        PN_next(y_index, x_index) = 0.25 * (PN(y_index, x_index + 1) + PN(y_index, x_index - 1) + PN(y_index + 1, x_index) + PN(y_index - 1, x_index)) - dt * ((PNU(y_index, x_index + 1) - PNU(y_index, x_index - 1)) / (2 * dx) + (PNV(y_index + 1, x_index) - PNV(y_index - 1, x_index)) / (2 * dx));
    }
}

__global__ void time_step(real_t *velocity_x, real_t *velocity_y,
                          real_t *acceleration_x, real_t *acceleration_y,
                          real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
                          real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
                          real_t *mass_velocity, real_t *mass_0, real_t *mass_1, real_t dx, real_t dt, int_t N)
{
    grid_group whole_grid = this_grid();

    int x_index = threadIdx.x + blockIdx.x * blockDim.x;
    int y_index = threadIdx.y + blockIdx.y * blockDim.y;

    boundary_condition(mass_0, 1, N, x_index, y_index);
    boundary_condition(mass_velocity_x_0, -1, N, x_index, y_index);
    boundary_condition(mass_velocity_y_0, -1, N, x_index, y_index);

    if ((1 <= x_index) && (x_index <= N) && (1 <= y_index) && (y_index <= N))
    {
        U(y_index, x_index) = PNU(y_index, x_index) / PN(y_index, x_index);
        V(y_index, x_index) = PNV(y_index, x_index) / PN(y_index, x_index);
        PNUV(y_index, x_index) = PN(y_index, x_index) * U(y_index, x_index) * V(y_index, x_index);
    }

    if ((x_index <= N + 1) && (y_index <= N + 1))
    {
        DU(y_index, x_index) = PN(y_index, x_index) * U(y_index, x_index) * U(y_index, x_index) + 0.5 * gravity * (PN(y_index, x_index) * PN(y_index, x_index) / density);
        DV(y_index, x_index) = PN(y_index, x_index) * V(y_index, x_index) * V(y_index, x_index) + 0.5 * gravity * (PN(y_index, x_index) * PN(y_index, x_index) / density);
    }

    whole_grid.sync();

    if ((1 <= x_index) && (x_index <= N) && (1 <= y_index) && (y_index <= N))
    {
        PNU_next(y_index, x_index) = 0.5 * (PNU(y_index, x_index + 1) + PNU(y_index, x_index - 1)) - dt * ((DU(y_index, x_index + 1) - DU(y_index, x_index - 1)) / (2 * dx) + (PNUV(y_index, x_index + 1) - PNUV(y_index, x_index - 1)) / (2 * dx));
        PNV_next(y_index, x_index) = 0.5 * (PNV(y_index + 1, x_index) + PNV(y_index - 1, x_index)) - dt * ((DV(y_index + 1, x_index) - DV(y_index - 1, x_index)) / (2 * dx) + (PNUV(y_index + 1, x_index) - PNUV(y_index - 1, x_index)) / (2 * dx));
        PN_next(y_index, x_index) = 0.25 * (PN(y_index, x_index + 1) + PN(y_index, x_index - 1) + PN(y_index + 1, x_index) + PN(y_index - 1, x_index)) - dt * ((PNU(y_index, x_index + 1) - PNU(y_index, x_index - 1)) / (2 * dx) + (PNV(y_index + 1, x_index) - PNV(y_index - 1, x_index)) / (2 * dx));
    }
}

// TODO: Rewrite boundary_condition as a device function.
__device__ void boundary_condition(real_t *domain_variable, int sign, int_t N, int x_index, int y_index)
{
#define VAR(y, x) domain_variable[(y) * (N + 2) + (x)]

    if (x_index == 0 && y_index == 0)
    {
        VAR(0, 0) = sign * VAR(2, 2);
    }

    if (x_index == 0 && y_index == N + 1)
    {
        VAR(N + 1, 0) = sign * VAR(N - 1, 2);
    }

    if (x_index == N + 1 && y_index == 0)
    {
        VAR(0, N + 1) = sign * VAR(2, N - 1);
    }

    if (x_index == N + 1 && y_index == N + 1)
    {
        VAR(N + 1, N + 1) = sign * VAR(N - 1, N - 1);
    }

    if (x_index == 2 && 1 <= y_index && y_index <= N)
    {
        VAR(y_index, 0) = sign * VAR(y_index, x_index);
    }
    if (x_index == N - 1 && 1 <= y_index && y_index <= N)
    {
        VAR(y_index, N + 1) = sign * VAR(y_index, x_index);
    }
    if (y_index == 2 && 1 <= x_index && x_index <= N)
    {
        VAR(0, x_index) = sign * VAR(y_index, x_index);
    }
    if (y_index == N - 1 && 1 <= x_index && x_index <= N)
    {
        VAR(N + 1, x_index) = sign * VAR(y_index, x_index);
    }
#undef VAR
}

void domain_init(void)
{
    int elements = (N + 2) * (N + 2);
    array_size = elements * sizeof(real_t);

    // TODO: Allocate device buffers for masses, velocities and accelerations.
    // -----------------------------------------------------
    h_mass_0 = (real_t *)calloc(elements, sizeof(real_t));
    h_mass_1 = (real_t *)calloc(elements, sizeof(real_t));
    cudaMalloc(&d_mass_0, array_size);
    cudaMalloc(&d_mass_1, array_size);

    h_mass_velocity_x_0 = (real_t *)calloc(elements, sizeof(real_t));
    h_mass_velocity_x_1 = (real_t *)calloc(elements, sizeof(real_t));
    h_mass_velocity_y_0 = (real_t *)calloc(elements, sizeof(real_t));
    h_mass_velocity_y_1 = (real_t *)calloc(elements, sizeof(real_t));
    cudaMalloc(&d_mass_velocity_x_0, array_size);
    cudaMalloc(&d_mass_velocity_x_1, array_size);
    cudaMalloc(&d_mass_velocity_y_0, array_size);
    cudaMalloc(&d_mass_velocity_y_1, array_size);

    h_mass_velocity = (real_t *)calloc(elements, sizeof(real_t));
    cudaMalloc(&d_mass_velocity, array_size);

    h_velocity_x = (real_t *)calloc(elements, sizeof(real_t));
    h_velocity_y = (real_t *)calloc(elements, sizeof(real_t));
    h_acceleration_x = (real_t *)calloc(elements, sizeof(real_t));
    h_acceleration_y = (real_t *)calloc(elements, sizeof(real_t));
    cudaMalloc(&d_velocity_x, array_size);
    cudaMalloc(&d_velocity_y, array_size);
    cudaMalloc(&d_acceleration_x, array_size);
    cudaMalloc(&d_acceleration_y, array_size);

    for (int_t y = 1; y <= N; y++)
    {
        for (int_t x = 1; x <= N; x++)
        {
            h_mass_0[y * (N + 2) + x] = 1e-3;
            h_mass_velocity_x_0[y * (N + 2) + x] = 0.0;
            h_mass_velocity_y_0[y * (N + 2) + x] = 0.0;

            real_t cx = x - N / 2;
            real_t cy = y - N / 2;
            if (sqrt(cx * cx + cy * cy) < N / 20.0)
            {
                h_mass_0[y * (N + 2) + x] -= 5e-4 * exp(
                                                        -4 * pow(cx, 2.0) / (real_t)(N)-4 * pow(cy, 2.0) / (real_t)(N));
            }

            h_mass_0[y * (N + 2) + x] *= density;
        }
    }

    dx = domain_size / (real_t)N;
    dt = 5e-2;

    cudaMemcpy(d_mass_0, h_mass_0, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_1, h_mass_1, array_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mass_velocity_x_0, h_mass_velocity_x_0, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_velocity_x_1, h_mass_velocity_x_1, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_velocity_y_0, h_mass_velocity_y_0, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_velocity_y_1, h_mass_velocity_y_1, array_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_mass_velocity, h_mass_velocity, array_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_velocity_x, h_velocity_x, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_y, h_velocity_y, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceleration_x, h_acceleration_x, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceleration_y, h_acceleration_y, array_size, cudaMemcpyHostToDevice);
}

void domain_save(int_t iteration)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    FILE *out = fopen(filename, "wb");
    if (!out)
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(1);
    }
    // fwrite ( mass[0], (N+2)*(N+2), sizeof(real_t), out );
    for (int_t y = 1; y <= N; y++)
    {
        fwrite(&h_mass_0[y * (N + 2) + 1], N, sizeof(real_t), out);
    }
    fclose(out);
}

void domain_finalize(void)
{
    free(h_mass_0);
    free(h_mass_1);
    free(h_mass_velocity_x_0);
    free(h_mass_velocity_x_1);
    free(h_mass_velocity_y_0);
    free(h_mass_velocity_y_1);
    free(h_mass_velocity);
    free(h_velocity_x);
    free(h_velocity_y);
    free(h_acceleration_x);
    free(h_acceleration_y);

    // TODO: Free device arrays
    cudaFree(d_mass_0);
    cudaFree(d_mass_1);
    cudaFree(d_mass_velocity_x_0);
    cudaFree(d_mass_velocity_x_1);
    cudaFree(d_mass_velocity_y_0);
    cudaFree(d_mass_velocity_y_1);
    cudaFree(d_mass_velocity);
    cudaFree(d_velocity_x);
    cudaFree(d_velocity_y);
    cudaFree(d_acceleration_x);
    cudaFree(d_acceleration_y);
}
