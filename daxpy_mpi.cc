/* Copyright 2018 Stanford University, NVIDIA Corporation
*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

////////////////////////////////////////////////////////////
// THIS EXAMPLE MUST BE BUILT WITH A VERSION
// OF GASNET CONFIGURED WITH MPI COMPATIBILITY
//
// NOTE THAT GASNET ONLY SUPPORTS MPI-COMPATIBILITY
// ON SOME CONDUITS. CURRENTLY THESE ARE IBV, GEMINI,
// ARIES, MXM, and OFI. IF YOU WOULD LIKE ADDITIONAL
// CONDUITS SUPPORTED PLEASE CONTACT THE MAINTAINERS
// OF GASNET.
//
// Note: there is a way to use this example with the
// MPI conduit, but you have to have a version of 
// MPI that supports MPI_THREAD_MULTIPLE. See the 
// macro GASNET_CONDUIT_MPI below.
////////////////////////////////////////////////////////////

#define _GNU_SOURCE
#define GEO
#define USE_AVX
// #define LEAF_COMM
// #define MID_COMM

#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include <ctime>
#include <unistd.h>
#include <sched.h>
#ifdef USE_AVX
#include <x86intrin.h>
#endif
// Need MPI header file
#ifdef GEO
#include <geopm.h>
#endif
// #include "mpi.h"
#include <mpi.h>

#include "legion.h"
#include "msr_core.h"
#include "master.h"

//#include "daxpy_mapper.h"

#define TASK_NAME_SIZE 256
#define NITER 256
#define POW_IMBALANCE 0.5
typedef double vecd4 __attribute__((vector_size (32)));

using namespace Legion;

enum TaskID
{
	TOP_LEVEL_TASK_ID,
	MPI_INTEROP_TASK_ID,
	DAXPY_TASK_ID_VEC,
	DAXPY_TASK_ID_SLEEP,
	INIT_FIELD_TASK_ID,
	CHECK_TASK_ID,
	FINISH_TASK_ID,
};

enum
{
	FID_X,
	FID_Y,
	FID_Z1,
	FID_Z2,
};

#ifdef GEO
struct daxpy_arg
{
	double alpha;
	uint64_t rid;
};

// struct mpi_int_arg
// {
// 	int size;
// 	uint64_t rid;
// };
#endif

// Here is our global MPI-Legion handshake
// You can have as many of these as you 
// want but the common case is just to
// have one per Legion-MPI rank pair
MPILegionHandshake handshake;

// Have a global static number of iterations for
// this example, but you can easily configure it
// from command line arguments which get passed 
// to both MPI and Legion
const int total_iterations = 32;

void set_rapl(unsigned sec, double watts, double pu, double su, unsigned affinity)
{
	uint64_t power = (unsigned long) (watts / pu);
	uint64_t seconds;
	uint64_t timeval_y = 0, timeval_x = 0;
	double logremainder = 0;

	timeval_y = (uint64_t) log2(sec / su);
	// store the mantissa of the log2
	logremainder = (double) log2(sec / su) - (double) timeval_y;
	timeval_x = 0;
	// based on the mantissa, we can choose the appropriate multiplier
	if (logremainder > 0.15 && logremainder <= 0.45)
	{
	    timeval_x = 1;
	}
	else if (logremainder > 0.45 && logremainder <= 0.7)
	{
	    timeval_x = 2;
	}
	else if (logremainder > 0.7)
	{
	    timeval_x = 3;
	}
	// store the bits in the Intel specified format
	seconds = (uint64_t) (timeval_y | (timeval_x << 5));
	uint64_t rapl = 0x0 | power | (seconds << 17);

	rapl |= (1LL << 15) | (1LL << 16);
	write_msr_by_coord(affinity, 0, 0, MSR_PKG_POWER_LIMIT, rapl);
}

void finish_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("Finish task execute on rank %d\n", rank);
	// fflush(stdout);
	// handshake.legion_handoff_to_mpi();
	MPI_Barrier(MPI_COMM_WORLD);
}

void init_field_task(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
{
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	assert(regions.size() == 1); 
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 1);

#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	assert(task->arglen == sizeof(uint64_t));
	uint64_t rid = *((uint64_t*)task->args);
	int err = geopm_prof_enter(rid);
	if (err)
	{
		fprintf(stderr, "Geo error (enter) in init task\n");
	}
	// handshake.legion_wait_on_mpi();
#endif

	FieldID fid = *(task->regions[0].privilege_fields.begin());
	const int point = task->index_point.point_data[0];
	// printf("Initializing field %d for block %d...\n", fid, point);

	// const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
	FieldAccessor<WRITE_DISCARD, double, 1, coord_t, 
		Realm::AffineAccessor<double, 1, coord_t>, false > acc(regions[0], fid);
	// Note here that we get the domain for the subregion for
	// this task from the runtime which makes it safe for running
	// both as a single task and as part of an index space of tasks.
	Rect<1> rect = runtime->get_index_space_domain(ctx,
			task->regions[0].region.get_index_space());
	for (PointInRectIterator<1> pir(rect); pir(); pir++)
	{
		acc[*pir] = drand48();
	}

#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	err = geopm_prof_exit(rid);
	if (err)
	{
		fprintf(stderr, "Geo error (exit) in init task\n");
	}
	// handshake.legion_wait_on_mpi();
#endif
	gettimeofday(&t2, NULL);
	printf("init task %d execution time: %lf\n", point, (t2.tv_sec - t1.tv_sec) + 
			(t2.tv_usec - t1.tv_usec) / 1000000.0);
}

// this daxpy task does 100 iterations using vector instructions
// high power
void daxpy_task_vec(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
{
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	assert(regions.size() == 2);
	assert(task->regions.size() == 2);

	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("daxpy task 100 point %d executing on rank %d\n", 
			// (int) task->index_point.point_data[0], rank);
	
#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	assert(task->arglen == sizeof(struct daxpy_arg));
	int err = 0;
	struct daxpy_arg *darg = (daxpy_arg*)task->args;
	const double alpha = darg->alpha;
	const uint64_t rid = darg->rid; 
	err = geopm_prof_enter(rid);
	if (err)
	{
		fprintf(stderr, "Geo error (enter) in daxpy task vec\n");
	}
	// handshake.legion_wait_on_mpi();
#else
	assert(task->arglen == sizeof(double));
	const double alpha = *((const double*)task->args);
#endif
	const int point = task->index_point.point_data[0];
	// printf("Running daxpy computation with alpha %.8g for point %d...\n", 
			// alpha, point);

#ifndef USE_AVX
	const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
	const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
	const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z1);
#else
	FieldAccessor<READ_ONLY, double, 1, coord_t, 
		Realm::AffineAccessor<double, 1, coord_t>, false > acc_x(regions[0], FID_X);
	FieldAccessor<READ_ONLY, double, 1, coord_t, 
		Realm::AffineAccessor<double, 1, coord_t>, false > acc_y(regions[0], FID_Y);
	FieldAccessor<WRITE_DISCARD, double, 1, coord_t, 
		Realm::AffineAccessor<double, 1, coord_t>, false > acc_z(regions[1], FID_Z1);
#endif

	for (int i = 0; i < 1000; i++)
	{
		Rect<1> rect = runtime->get_index_space_domain(ctx,
				task->regions[0].region.get_index_space());
#ifndef USE_AVX
		for (PointInRectIterator<1> pir(rect); pir(); pir++)
		{
			acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
		}
#else
		__m256d alphavec = _mm256_set_pd(alpha, alpha, alpha, alpha);
		for (PointInRectIterator<1> pir(rect); pir(); pir++, pir++, pir++, pir++)
		{
			// double xvals[4] = {acc_x[*pir], acc_x[*pir + 1], acc_x[*pir + 2], acc_x[*pir + 3]};
			// double yvals[4] = {acc_y[*pir], acc_y[*pir + 1], acc_y[*pir + 2], acc_y[*pir + 3]}; 
			__m256d xvec = _mm256_load_pd(acc_x.ptr(*pir));
			__m256d yvec = _mm256_load_pd(acc_y.ptr(*pir));

			// __m256d xvec = _mm256_set_pd(xvals[0], xvals[1], xvals[2], xvals[3]);
			// __m256d yvec = _mm256_set_pd(yvals[0], yvals[1], yvals[2], yvals[3]);
			__m256d res = _mm256_mul_pd(xvec, yvec);
			res = _mm256_mul_pd(res, alphavec);
			_mm256_stream_pd(acc_z.ptr(*pir), res);
		}
#endif
	}
#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	err = geopm_prof_exit(rid);
	if (err)
	{
		fprintf(stderr, "Geo error (exit) in daxpy task vec\n");
	}
	// handshake.legion_wait_on_mpi();
#endif
	gettimeofday(&t2, NULL);
	printf("daxpy task AVX #%d <node %d> execution time: %lf\n", point, rank, (t2.tv_sec - t1.tv_sec) + 
			(t2.tv_usec - t1.tv_usec) / 1000000.0);
}

// this daxpy task does 1000 iterations with sleeps inserted
// low power
void daxpy_task_sleep(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
{
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	assert(regions.size() == 2);
	assert(task->regions.size() == 2);

	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("daxpy task 1000 point %d executing on rank %d\n", 
			// (int) task->index_point.point_data[0], rank);
	
#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	assert(task->arglen == sizeof(struct daxpy_arg));
	int err = 0;
	struct daxpy_arg *darg = (daxpy_arg*)task->args;
	const double alpha = darg->alpha;
	const uint64_t rid = darg->rid;
	err = geopm_prof_enter(rid);
	if (err)
	{
		fprintf(stderr, "Geo error (enter) in daxpy task sleep\n");
	}
	// handshake.legion_wait_on_mpi();
#else
	assert(task->arglen == sizeof(double));
	const double alpha = *((const double*)task->args);
#endif
	const int point = task->index_point.point_data[0];
	// printf("Running daxpy 1000x computation with alpha %.8g for point %d...\n", 
			// alpha, point);

	// const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
	// const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
	// const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z2);
	FieldAccessor<READ_ONLY, double, 1, coord_t, 
			Realm::AffineAccessor<double, 1, coord_t>, false > acc_x(regions[0], FID_X);
	FieldAccessor<READ_ONLY, double, 1, coord_t, 
			Realm::AffineAccessor<double, 1, coord_t>, false > acc_y(regions[0], FID_Y);
	FieldAccessor<WRITE_DISCARD, double, 1, coord_t, 
			Realm::AffineAccessor<double, 1, coord_t>, false > acc_z(regions[1], FID_Z2);

	for (int i = 0; i < 1000; i++)
	{
		Rect<1> rect = runtime->get_index_space_domain(ctx,
				task->regions[0].region.get_index_space());
		for (PointInRectIterator<1> pir(rect); pir(); pir++)
		{
			acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
			sleep(0.1);
		}
	}
#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	err = geopm_prof_exit(rid);
	if (err)
	{
		fprintf(stderr, "Geo error (exit) in daxpy task sleep\n");
	}
	// handshake.legion_wait_on_mpi();
#endif
	gettimeofday(&t2, NULL);
	printf("daxpy task IO #%d <node %d> execution time: %lf\n", point, rank, (t2.tv_sec - t1.tv_sec) + 
			(t2.tv_usec - t1.tv_usec) / 1000000.0);
}

void check_task(const Task *task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime)
{
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	assert(regions.size() == 2);
	assert(task->regions.size() == 2);

#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	assert(task->arglen == sizeof(struct daxpy_arg));
	struct daxpy_arg *darg = (daxpy_arg*)task->args;
	int err = geopm_prof_enter(darg->rid);
	const double alpha = darg->alpha;
	if (err)
	{
		fprintf(stderr, "Geo error (enter) in check task\n");
	}
	// handshake.legion_wait_on_mpi();
#else
	assert(task->arglen == sizeof(double));
	const double alpha = *((const double*)task->args);
#endif

	const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
	const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
	const FieldAccessor<READ_ONLY,double,1> acc_z(regions[1], FID_Z1);
	// FieldAccessor<READ_ONLY, double, 1, coord_t, 
	// 	Realm::AffineAccessor<double, 1, coord_t> > acc_x(regions[0], FID_X);
	// FieldAccessor<READ_ONLY, double, 1, coord_t, 
	// 	Realm::AffineAccessor<double, 1, coord_t> > acc_y(regions[0], FID_Y);
	// FieldAccessor<READ_ONLY, double, 1, coord_t, 
	// 	Realm::AffineAccessor<double, 1, coord_t> > acc_z(regions[1], FID_Z1);

	printf("Checking results...");
	Rect<1> rect = runtime->get_index_space_domain(ctx,
			task->regions[0].region.get_index_space());
	bool all_passed = true;
	for (PointInRectIterator<1> pir(rect); pir(); pir++)
	{
		// double a = acc_x[*pir];
		// double b = acc_y[*pir];
		// double c;
		// for (int i = 0; i < NITER; i++)
		// {
		// 	c = c + alpha * a + alpha * b + b * a;
		// }
		// double expected = c;
		double expected = alpha * acc_x[*pir] + acc_y[*pir];
		double received = acc_z[*pir];
		// Probably shouldn't check for floating point equivalence but
		// the order of operations are the same should they should
		// be bitwise equal.
		if (expected != received)
			all_passed = false;
	}
	if (all_passed)
		printf("SUCCESS!\n");
	else
		printf("FAILURE!\n");
	// fflush(stdout);

#ifdef GEO
	// handshake.legion_handoff_to_mpi();
	err = geopm_prof_exit(darg->rid);
	if (err)
	{
		fprintf(stderr, "Geo error (exit) in check task\n");
	}
	// handshake.legion_wait_on_mpi();
#endif
	gettimeofday(&t2, NULL);
	printf("check task execution time: %lf\n", (t2.tv_sec - t1.tv_sec) + 
			(t2.tv_usec - t1.tv_usec) / 1000000.0);
}

void mpi_interop_task(const Task *task, 
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == -1)
	{
		printf("ERROR: invalid rank\n");
	}
	// printf("Hello from Legion MPI-Interop Task %lld (rank %d)\n", task->index_point[0], rank);

#ifdef GEO
	// uint64_t legion_rid;
	uint64_t init_rid;
	uint64_t daxpy_vec_rid;
	uint64_t daxpy_sleep_rid;
	uint64_t check_rid;
	int err;
	err = geopm_prof_epoch();
	if (err)
	{
		fprintf(stderr, "Geo error in mpi interop task\n");
	}
	char daxpy_str[TASK_NAME_SIZE];
	snprintf(daxpy_str, TASK_NAME_SIZE, "daxpy_child_task");
	err = geopm_prof_region(daxpy_str, GEOPM_REGION_HINT_COMPUTE, &daxpy_vec_rid);
	if (err)
	{
		fprintf(stderr, "Geo error in mpi interop task\n");
	}
	char daxpy10_str[TASK_NAME_SIZE];
	snprintf(daxpy10_str, TASK_NAME_SIZE, "daxpy10_child_task");
	err = geopm_prof_region(daxpy10_str, GEOPM_REGION_HINT_IO, &daxpy_sleep_rid);
	if (err)
	{
		fprintf(stderr, "Geo error in mpi interop task\n");
	}
	char init_str[TASK_NAME_SIZE];
	snprintf(init_str, TASK_NAME_SIZE, "init_child_task");
	err = geopm_prof_region(init_str, GEOPM_REGION_HINT_MEMORY, &init_rid);
	if (err)
	{
		fprintf(stderr, "Geo error in mpi interop task\n");
	}
	char check_str[TASK_NAME_SIZE];
	snprintf(check_str, TASK_NAME_SIZE, "check_child_task");
	err = geopm_prof_region(check_str, GEOPM_REGION_HINT_MEMORY, &check_rid);
	if (err)
	{
		fprintf(stderr, "Geo error in mpi interop task\n");
	}
	// char legion_str[TASK_NAME_SIZE];
	// snprintf(legion_str, TASK_NAME_SIZE, "legion_child_task");
	// err = geopm_prof_region(legion_str, GEOPM_REGION_HINT_COMPUTE, &legion_rid);
	// if (err)
	// {
	// 	fprintf(stderr, "Geo error in mpi interop task\n");
	// }
	// err = geopm_prof_enter(legion_rid);
	// if (err)
	// {
	// 	fprintf(stderr, "Geo error in mpi interop task\n");
	// }
	// printf("rank %d legion task rid %llu\n", rank, legion_rid);
	// printf("rank %d init task rid %lu\n", rank, init_rid);
	// printf("rank %d daxpy task rid %lu\n", rank, daxpy_rid);
	// printf("rank %d check task rid %lu\n", rank, check_rid);
#endif

	// handshake.legion_wait_on_mpi();
	IndexLauncher init_tl1;
	IndexLauncher init_tl2;
	IndexLauncher daxpy_tl;
	TaskLauncher check_tl;
	if (rank == 0)
	{
		// printf("rank %d tag %d\n", rank, 0);
#ifdef MID_COMM
		int count = 0;
		int recount;
		MPI_Status stat;
		// printf("rank %d tag %d\n", rank, count);
		MPI_Send(&count, 1, MPI_INT, 1, count, MPI_COMM_WORLD);
		// printf("rank %d sent %d\n", rank, count);
		MPI_Recv(&recount, 1, MPI_INT, 1, count, MPI_COMM_WORLD, &stat);
		printf("rank %d recv %d\n", rank, recount);
		count++;
#endif

		const InputArgs &command_args = Runtime::get_input_args();
		int mult = atoi(command_args.argv[2]);
		int num_elements = 0x1 << 24;
		int num_subregions = 1;
		if (mult > 0)
		{
			// TODO: disabled rank multiplier
			// num_subregions = (*((int *) task->args)) * mult;
			num_subregions = mult;
		}
		else if (mult < 0)
		{
			num_subregions = (*((int *) task->args)) / (mult * -1);
		}
		// printf("Running daxpy for %d elements...\n", num_elements);
		printf("Partitioning data into %d sub-regions...\n", num_subregions);

		// Create our logical regions using the same schemas as earlier examples
		Rect<1> elem_rect(0,num_elements-1);
		IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
		runtime->attach_name(is, "is");
		FieldSpace input_fs = runtime->create_field_space(ctx);
		runtime->attach_name(input_fs, "input_fs");
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx, input_fs);
			allocator.allocate_field(sizeof(double),FID_X);
			runtime->attach_name(input_fs, FID_X, "X");
			allocator.allocate_field(sizeof(double),FID_Y);
			runtime->attach_name(input_fs, FID_Y, "Y");
		}
		FieldSpace output_fs = runtime->create_field_space(ctx);
		runtime->attach_name(output_fs, "output_fs");
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx, output_fs);
			allocator.allocate_field(sizeof(double),FID_Z1);
			runtime->attach_name(output_fs, FID_Z1, "Z1");
			allocator.allocate_field(sizeof(double), FID_Z2);
			runtime->attach_name(output_fs, FID_Z2, "Z2");
		}
		// printf("creating regions\n");
		LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
		runtime->attach_name(input_lr, "input_lr");
		LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);
		runtime->attach_name(output_lr, "output_lr");

		// printf("creating partitions\n");
		Rect<1> color_bounds(0,num_subregions-1);
		IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
		IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
		runtime->attach_name(ip, "ip");

		LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
		runtime->attach_name(input_lp, "input_lp");
		LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
		runtime->attach_name(output_lp, "output_lp");

		ArgumentMap arg_map;

		// printf("creating launchers\n");
#ifdef GEO
		IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
				TaskArgument(&init_rid, sizeof(uint64_t)), arg_map);
#else
		IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
				TaskArgument(NULL, 0), arg_map);
#endif
		
		init_launcher.add_region_requirement(
				RegionRequirement(input_lp, 0/*projection ID*/, 
				WRITE_DISCARD, EXCLUSIVE, input_lr));
		init_launcher.region_requirements[0].add_field(FID_X);
		// runtime->execute_index_space(ctx, init_launcher);
		init_tl1 = init_launcher;
		// MPI_Send(&init_tl1, sizeof(init_tl1), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		runtime->execute_index_space(ctx, init_tl1);

		init_launcher.region_requirements[0].privilege_fields.clear();
		init_launcher.region_requirements[0].instance_fields.clear();
		init_launcher.region_requirements[0].add_field(FID_Y);
		// runtime->execute_index_space(ctx, init_launcher);
		init_tl2 = init_launcher;
		runtime->execute_index_space(ctx, init_tl2);
		// MPI_Send(&init_tl2, sizeof(init_tl2), MPI_CHAR, 1, 1, MPI_COMM_WORLD);

		const double alpha = drand48();
#ifdef GEO
		struct daxpy_arg darg;
		darg.alpha = alpha;
		darg.rid = daxpy_vec_rid;
		IndexLauncher daxpy_launcher_vec(DAXPY_TASK_ID_VEC, color_is,
				TaskArgument(&darg, sizeof(struct daxpy_arg)), arg_map);
		darg.rid = daxpy_sleep_rid;
		IndexLauncher daxpy_launcher_sleep(DAXPY_TASK_ID_SLEEP, color_is,
				TaskArgument(&darg, sizeof(struct daxpy_arg)), arg_map);
#else
		IndexLauncher daxpy_launcher_vec(DAXPY_TASK_ID_VEC, color_is,
				TaskArgument(&alpha, sizeof(double)), arg_map);
		IndexLauncher daxpy_launcher_sleep(DAXPY_TASK_ID_SLEEP, color_is,
				TaskArgument(&alpha, sizeof(double)), arg_map);
#endif
		// the 100x daxpy
		daxpy_launcher_vec.add_region_requirement(
				RegionRequirement(input_lp, 0/*projection ID*/,
				READ_ONLY, SIMULTANEOUS, input_lr));
		daxpy_launcher_vec.region_requirements[0].add_field(FID_X);
		daxpy_launcher_vec.region_requirements[0].add_field(FID_Y);
		daxpy_launcher_vec.add_region_requirement(
				RegionRequirement(output_lp, 0/*projection ID*/,
				WRITE_DISCARD, RELAXED, output_lr));
		daxpy_launcher_vec.region_requirements[1].add_field(FID_Z1);
		// the 1000x daxpy
		daxpy_launcher_sleep.add_region_requirement(
				RegionRequirement(input_lp, 0/*projection ID*/,
				READ_ONLY, SIMULTANEOUS, input_lr));
		daxpy_launcher_sleep.region_requirements[0].add_field(FID_X);
		daxpy_launcher_sleep.region_requirements[0].add_field(FID_Y);
		daxpy_launcher_sleep.add_region_requirement(
				RegionRequirement(output_lp, 0/*projection ID*/,
				WRITE_DISCARD, RELAXED, output_lr));
		daxpy_launcher_sleep.region_requirements[1].add_field(FID_Z2);

		// runtime->execute_index_space(ctx, daxpy_launcher_vec);
		// daxpy_tl = daxpy_launcher_vec;
		// runtime->execute_index_space(ctx, daxpy_tl);
		// MPI_Send(&daxpy_tl, sizeof(daxpy_tl), MPI_CHAR, 1, 2, MPI_COMM_WORLD);

		IndexLauncher finish_launcher(FINISH_TASK_ID, color_is,
				TaskArgument(NULL, 0), arg_map);
		
		FutureMap fm = runtime->execute_index_space(ctx, daxpy_launcher_vec);
		FutureMap fm2 = runtime->execute_index_space(ctx, daxpy_launcher_sleep);
		// FutureMap fm[total_iterations];
		// for (int i = 0; i < total_iterations; i++)
		// {
			// printf("rank %d itr %d EXECUTE\n", rank, i);
			// fm[i] = runtime->execute_index_space(ctx, daxpy_tl);
			// fm[i].wait_all_results();
			// printf("rank %d itr %d RESULTS\n", rank, i);
			// sleep(0.2);
			// printf("rank %d got results\n", rank);
		// }

		// printf("rank %d CHECK\n", rank);
#ifdef GEO
		darg.rid = check_rid;
		TaskLauncher check_launcher(CHECK_TASK_ID, 
				TaskArgument(&darg, sizeof(struct daxpy_arg)));
#else
		TaskLauncher check_launcher(CHECK_TASK_ID, 
				TaskArgument(&alpha, sizeof(double)));
#endif
		// check_launcher.add_region_requirement(
		// 		RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
		// check_launcher.region_requirements[0].add_field(FID_X);
		// check_launcher.region_requirements[0].add_field(FID_Y);
		// check_launcher.add_region_requirement(
		// 		RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
		// check_launcher.region_requirements[1].add_field(FID_Z1);
		// Future fcheck = runtime->execute_task(ctx, check_launcher);
		// MPI_Send(&check_tl, sizeof(check_tl), MPI_CHAR, 1, 3, MPI_COMM_WORLD);

		// printf("rank %d freeing\n", rank);
		runtime->destroy_logical_region(ctx, input_lr);
		runtime->destroy_logical_region(ctx, output_lr);
		runtime->destroy_logical_region(ctx, output_lr);
		runtime->destroy_field_space(ctx, input_fs);
		runtime->destroy_field_space(ctx, output_fs);
		runtime->destroy_field_space(ctx, output_fs);
		runtime->destroy_index_space(ctx, is);

		fm.wait_all_results();
		fm2.wait_all_results();
		// for (int i = 0; i < total_iterations; i++)
		// {
			// fm[i].wait_all_results();
		// }
		// fcheck.wait();

		// runtime->execute_index_space(ctx, finish_launcher);

		// handshake.legion_handoff_to_mpi();
		// printf("rank %d complete\n", rank);
	}
	else
	{
		for (int i = 0; i < total_iterations; i++)
		{
			// printf("rank %d itr %d LEGION\n", rank, i);
			// handshake.legion_wait_on_mpi();
			// printf("rank %d itr %d MPI\n", rank, i);
			// sleep(0.2);
			// handshake.legion_handoff_to_mpi();
		}
		// printf("rank %d complete\n", rank);
		// handshake.legion_handoff_to_mpi();
	}
#ifdef GEO
	// err = geopm_prof_exit(legion_rid);
	// if (err)
	// {
	// 	fprintf(stderr, "Geo error in mpi interop\n");
	// }
#endif
}

void top_level_task(const Task *task, 
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	// Both the application and Legion mappers have access to
	// the mappings between MPI Ranks and Legion address spaces
	// The reverse mapping goes the other way
	const std::map<int,AddressSpace> &forward_mapping = 
	runtime->find_forward_MPI_mapping();
	for (std::map<int,AddressSpace>::const_iterator it = 
		forward_mapping.begin(); it != forward_mapping.end(); it++)
	{
		printf("MPI Rank %d maps to Legion Address Space %d\n", it->first, it->second);
	}

	int rank = -1, size = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// printf("Hello from Legion Top-Level Task on rank %d\n", rank);
	// Do a must epoch launch to align with the number of MPI ranks
	MustEpochLauncher must_epoch_launcher;
	Rect<1> launch_bounds(0,size - 1);
	ArgumentMap args_map;
	IndexLauncher index_launcher(MPI_INTEROP_TASK_ID, launch_bounds, 
		TaskArgument(&size, sizeof(int)), args_map);
	must_epoch_launcher.add_index_task(index_launcher);
	runtime->execute_must_epoch(ctx, must_epoch_launcher);
}

int main(int argc, char **argv)
{
	printf("running legion\n");
#ifdef GASNET_CONDUIT_MPI
	// The GASNet MPI conduit requires special start-up
	// in order to handle MPI calls from multiple threads
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	// If you fail this assertion, then your version of MPI
	// does not support calls from multiple threads and you 
	// cannot use the GASNet MPI conduit
	if (provided < MPI_THREAD_MULTIPLE)
	printf("ERROR: Your implementation of MPI does not support "
		"MPI_THREAD_MULTIPLE which is required for use of the "
		"GASNet MPI conduit with the Legion-MPI Interop!\n");
	assert(provided == MPI_THREAD_MULTIPLE);
#else
	// Perform MPI start-up like normal for most GASNet conduits
	printf("mpi init\n");
	MPI_Init(&argc, &argv);
	printf("waiting\n");
#endif

	if (argc < 3)
	{
		printf("ERROR: daxpy_mpi <powlim> <partition multiplier>");
	}
	double limit = atof(argv[1]);

	int rank = -1, size = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("Hello from MPI process %d of %d\n", rank, size);

	// Configure the Legion runtime with the rank of this process
	Runtime::configure_MPI_interoperability(rank);
	// Register our task variants

	if (init_msr())
	{
		printf("ERROR: could not init libmsr\n");
	}

	uint64_t aperf_begin, mperf_begin;
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_begin);
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_begin);
	struct timeval t1;
	gettimeofday(&t1, NULL);

	int partitions = 1;
	int mult = atoi(argv[2]);
	if ( mult > 0)
	{
		partitions = size * mult;
	}
	else if (mult < 0)
	{
		partitions = size / (atoi(argv[2]) * -1);
	}
	printf("There will be %d partitions\n", partitions);

	uint64_t rapl_bits;
	uint64_t unit;
	read_msr_by_coord(0, 0, 0, MSR_RAPL_POWER_UNIT, &unit);
	uint64_t power_unit = unit & 0xF;
	double pu = 1.0 / (0x1 << power_unit);
	uint64_t seconds_unit_raw = (unit >> 16) & 0x1F;
	double energy_unit;
	double su = 1.0 / (0x1 << seconds_unit_raw);
	unsigned eu = (unit >> 8) & 0x1F;
	energy_unit = 1.0 / (0x1 << eu);

#ifndef GEO
	set_rapl(1, limit, pu, su, 0);
	set_rapl(1, limit, pu, su, 1);
#endif
	uint64_t energy_s1_begin;
	uint64_t energy_s2_begin;
	read_msr_by_coord(0, 0, 0, MSR_PKG_ENERGY_STATUS, &energy_s1_begin);
	read_msr_by_coord(1, 0, 0, MSR_PKG_ENERGY_STATUS, &energy_s2_begin);
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl_bits);
	printf("RAPL bits %lx\n", rapl_bits);

	{
		TaskVariantRegistrar top_level_registrar(TOP_LEVEL_TASK_ID);
		top_level_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(top_level_registrar, 
			"Top Level Task");
		Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
	}
	{
		TaskVariantRegistrar mpi_interop_registrar(MPI_INTEROP_TASK_ID);
		mpi_interop_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<mpi_interop_task>(mpi_interop_registrar,
			"MPI Interop Task");
	}
	{
		TaskVariantRegistrar init_task_registrar(INIT_FIELD_TASK_ID);
		init_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		init_task_registrar.set_leaf();
		Runtime::preregister_task_variant<init_field_task>(init_task_registrar,
			"Init Task");
	}
	{
		TaskVariantRegistrar daxpy_task_registrar(DAXPY_TASK_ID_VEC);
		daxpy_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		daxpy_task_registrar.set_leaf();
		Runtime::preregister_task_variant<daxpy_task_vec>(daxpy_task_registrar,
			"DAXPY Task Vec");
	}
	{
		TaskVariantRegistrar daxpy_task_registrar(DAXPY_TASK_ID_SLEEP);
		daxpy_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		daxpy_task_registrar.set_leaf();
		Runtime::preregister_task_variant<daxpy_task_sleep>(daxpy_task_registrar,
			"DAXPY Task Sleep");
	}
	{
		TaskVariantRegistrar check_task_registrar(CHECK_TASK_ID);
		check_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<check_task>(check_task_registrar,
			"Check Task");
	}

	// Runtime::add_registration_callback(mapper_registration);
	
	// {
	// 	TaskVariantRegistrar finish_task_registrar(FINISH_TASK_ID);
	// 	finish_task_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
	// 	Runtime::preregister_task_variant<finish_task>(finish_task_registrar,
	// 		"Finish Task");
	// }
	// Create a handshake for passing control between Legion and MPI
	// Indicate that MPI has initial control and that there is one
	// participant on each side
	handshake = Runtime::create_handshake(false/*MPI initial control*/,
		1/*MPI participants*/,
		21/*Legion participants*/);
	// Start the Legion runtime in background mode
	// This call will return immediately
	Runtime::start(argc, argv, false/*background*/);
	// Run your MPI program like normal
	// If you want strict bulk-synchronous execution include
	// the barriers protected by this variable, otherwise
	// you can elide them, they are not required for correctness
	// MPI_Barrier(MPI_COMM_WORLD);
	//
	// handshake.mpi_handoff_to_legion();

	// INIT TASK handshakes
	// there are two init tasks per partition
	// each init task has two Geo calls with the enter and exit handshakes
	for (int i = 0; i < partitions; i++)
	{
		// handshake.mpi_wait_on_legion(); // geo prof begin exit
		// handshake.mpi_handoff_to_legion(); // geo prof begin enter
		// handshake.mpi_wait_on_legion(); // geo prof end exit
		// handshake.mpi_handoff_to_legion(); // geo prof end enter
	}

	const bool strict_bulk_synchronous_execution = false;
	// for (int i = 0; i < total_iterations; i++)
	for (int i = 0; i < 1; i++)
	{
		// DAXPY TASK handshakes
		// one daxpy task executes per iteration
		// printf("rank %d handshake itr %d\n", rank, i);
		// handshake.mpi_wait_on_legion(); // geo prof begin exit
		if (strict_bulk_synchronous_execution) MPI_Barrier(MPI_COMM_WORLD);
		// printf("MPI rank %d geo enter %d\n", rank, i);
		// handshake.mpi_handoff_to_legion(); // geo prof begin enter
		// handshake.mpi_wait_on_legion(); // geo prof end exit
		if (strict_bulk_synchronous_execution) MPI_Barrier(MPI_COMM_WORLD);
		// printf("MPI rank %d geo exit %i\n", rank, i);
		// fflush(stdout);
		// handshake.mpi_handoff_to_legion(); // geo prof end enter
	}
	// printf("MPI rank %d finished daxpy\n", rank);

	// CHECK TASK handshakes
	// rank 0 is the only rank that checks
	if (rank == 0)
	{
		// handshake.mpi_wait_on_legion(); // geo prof begin exit
		// handshake.mpi_handoff_to_legion(); // geo prof begin enter
		// handshake.mpi_wait_on_legion(); // geo prof end exit
		// handshake.mpi_handoff_to_legion(); // geo prof end enter
	}
	// handshake.mpi_wait_on_legion(); // geo prof end exit
	// printf("MPI rank %d at barrier\n", rank);
	// MPI_Barrier(MPI_COMM_WORLD);

	// printf("MPI rank %d finished\n", rank);
	// When you're done wait for the Legion runtime to shutdown
	// Runtime::wait_for_shutdown();

	struct timeval t2;
	gettimeofday(&t2, NULL);
	double time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
#ifndef GEO
	set_rapl(1, 115.0, pu, su, 0);
	set_rapl(1, 115.0, pu, su, 1);
#endif
	uint64_t energy_s1_end;
	uint64_t energy_s2_end;
	read_msr_by_coord(0, 0, 0, MSR_PKG_ENERGY_STATUS, &energy_s1_end);
	read_msr_by_coord(1, 0, 0, MSR_PKG_ENERGY_STATUS, &energy_s2_end);
	read_msr_by_coord(0, 0, 0, MSR_PKG_POWER_LIMIT, &rapl_bits);
	
	unsigned long diffs1;
	unsigned long diffs2;	
	if (energy_s1_end < energy_s1_begin)
	{
		diffs1 = 0xFFFFFFFFul - energy_s1_begin + energy_s1_end;
	}
	else
	{
		diffs1 = energy_s1_end - energy_s1_begin;
	}
	if (energy_s2_end < energy_s2_begin)
	{
		diffs2 = 0xFFFFFFFFul - energy_s2_begin + energy_s2_end;
	}
	else
	{
		diffs2 = energy_s2_end - energy_s2_begin;
	}

	double pows1 = (double) diffs1 * energy_unit / time;
	double pows2 = (double) diffs2 * energy_unit / time;
	printf("MPI rank %d power s1: %lf s2: %lf\n", rank, pows1, pows2);
	// printf("RAPL bits %lx\n", rapl_bits);

	uint64_t aperf_end, mperf_end;
	read_msr_by_coord(0, 0, 0, MSR_IA32_APERF, &aperf_end);
	read_msr_by_coord(0, 0, 0, MSR_IA32_MPERF, &mperf_end);
	
	printf("MPI rank %d time %lf\n", rank, time);
	printf("MPI rank %d freq %lf\n", rank, (double) (aperf_end - aperf_begin) / (double) (mperf_end - mperf_begin) * 2.4);

	finalize_msr();
#ifndef GASNET_CONDUIT_MPI
	// Then finalize MPI like normal
	// Exception for the MPI conduit which does its own finalization
	MPI_Finalize();
#endif

	return 0;
}
