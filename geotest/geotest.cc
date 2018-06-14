#include "../mpi.h"
// #include <mpi.h>
#include <geopm.h>
#include <iostream>
#include <cstdlib>

// #define GEO

#define TASK_NAME_SIZE 256

typedef double vecd4 __attribute__((vector_size (32)));

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int size = -1, rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size < 1 || rank < 0)
	{
		std::cout << "MPI init failed\n";
		std::cout << "size " << size << std::endl;
		std::cout << "rank " << rank << std::endl;
	}

#ifdef GEO
	// Geo initq
	uint64_t t1rid;
	uint64_t t2rid;
	char str[TASK_NAME_SIZE];
	snprintf(str, TASK_NAME_SIZE, "task_%llu", 1lu);
	int err = geopm_prof_region(str, GEOPM_REGION_HINT_COMPUTE, &t1rid);
	if (err)
	{
		fprintf(stderr, "Geo error init\n");
	}
	snprintf(str, TASK_NAME_SIZE, "task_%llu", 2lu);
	err = geopm_prof_region(str, GEOPM_REGION_HINT_MEMORY, &t2rid);
	if (err)
	{
		fprintf(stderr, "Geo error init\n");
	}
	printf("rank %d has t1rid %llu\n", rank, t1rid);
	printf("rank %d has t1rid %llu\n", rank, t2rid);
#endif

	if (rank == 0)
	{
		err = geopm_prof_enter(t1rid);
		if (err)
		{
			fprintf(stderr, "Geo error  %d\n", rank);
		}
		double *a = (double *) malloc(2048 * sizeof(double));
		for (int i = 0; i < 2048; i++)
		{
			a[i] = 1.2 + (double) i / 4;
		}
		for (int i = 0; i < 2048; i += 4)
		{
			vecd4 v = *((vecd4 *) &a[i]);
			vecd4 *r = (vecd4 *) &a[i];
			vecd4 m = {i, i+1, i+2, i+3};
			*r = v * m;
			// a[i] += i * a[i];
		}
		err = geopm_prof_exit(t1rid);
		if (err)
		{
			fprintf(stderr, "Geo error  %d\n", rank);
		}
		std::cout << rank << ": " << a[0] << std::endl;
		std::cout << rank << ": " << a[1000] << std::endl;
	}
	else
	{
		err = geopm_prof_enter(t2rid);
		if (err)
		{
			fprintf(stderr, "Geo error  %d\n", rank);
		}
		double *a = (double *) malloc(2048 * sizeof(double));
		double *b = (double *) malloc(2048 * sizeof(double));
		double *c = (double *) malloc(2048 * sizeof(double));
		double *d = (double *) malloc(2048 * sizeof(double));
		double *e = (double *) malloc(2048 * sizeof(double));
		double *f = (double *) malloc(2048 * sizeof(double));
		double *g = (double *) malloc(2048 * sizeof(double));
		double *h = (double *) malloc(2048 * sizeof(double));
		double *l = (double *) malloc(2048 * sizeof(double));
		double *m = (double *) malloc(2048 * sizeof(double));
		for (int i = 0; i < 2048; i++)
		{
			a[i] = 0.5 + i;
			b[i] = 0.9 + i;
			c[i] = 1.5 + i;
			d[i] = 0.5 - i;
			e[i] = 0.9 - i;
			f[i] = 1.5 - i;
			g[i] = 2.1 + i;
			h[i] = 3.3 - i;
			l[i] = 4.6 + i;
			m[i] = 0.1 - i;
		}
		for (int i = 0; i < 2048; i++)
		{
			a[i] += b[i] + c[i] * d[i] / e[i] + f[i] * g[i] - h[i] + l[i] / m[i];
			b[i] += c[i] + d[i] * e[i] / f[i] + g[i] * h[i] - l[i] + m[i] / a[i];
			c[i] += d[i] + e[i] * f[i] / g[i] + h[i] * l[i] - m[i] + a[i] / b[i];
			d[i] += e[i] + f[i] * g[i] / h[i] + l[i] * m[i] - a[i] + b[i] / c[i];
			e[i] += f[i] + g[i] * h[i] / l[i] + m[i] * a[i] - b[i] + c[i] / d[i];
			f[i] += g[i] + h[i] * l[i] / m[i] + a[i] * b[i] - c[i] + d[i] / e[i];
			g[i] += h[i] + l[i] * m[i] / a[i] + b[i] * c[i] - d[i] + e[i] / f[i];
			h[i] += l[i] + m[i] * a[i] / b[i] + c[i] * d[i] - e[i] + f[i] / g[i];
			l[i] += m[i] + a[i] * b[i] / c[i] + d[i] * e[i] - f[i] + g[i] / h[i];
			m[i] += a[i] + b[i] * c[i] / d[i] + e[i] * f[i] - g[i] + h[i] / l[i];
		}
		err = geopm_prof_exit(t2rid);
		if (err)
		{
			fprintf(stderr, "Geo error  %d\n", rank);
		}
		std::cout << rank << ": " << a[0] << std::endl;
		std::cout << rank << ": " << b[0] << std::endl;
		std::cout << rank << ": " << c[0] << std::endl;
		std::cout << rank << ": " << d[0] << std::endl;
		std::cout << rank << ": " << e[0] << std::endl;
		std::cout << rank << ": " << f[0] << std::endl;
		std::cout << rank << ": " << g[0] << std::endl;
		std::cout << rank << ": " << h[0] << std::endl;
		std::cout << rank << ": " << l[0] << std::endl;
		std::cout << rank << ": " << m[0] << std::endl;
	}

	MPI_Finalize();
	return 0;
}
