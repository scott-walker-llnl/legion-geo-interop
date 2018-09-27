# Legion and GeoPM Iterop
This is part of the ECP power steering project. Intel's GeoPM power manager is designed to work with multi-process or multi-threaded programs, but Legion is a task based model. Legion and GeoPM are able to interoperate in certain configurations without significant changes to the Legion code. 

The standard setup for GeoPM is to create one process per compute node and then insert Geo markup calls inside the program to monitor. These markup calls use shared memory to communicate with the Geo process. From experimentation as well as discussions with the Legion developers, we were able to determine that it is safe to use shared memory communication without using Legion's "handshake" mechanism.

## Install Directions
Note: You should use the same compiler for all tools in the chain, if you compile GASNET with icc and Legion with gcc then the binaries will not link correctly. gcc 4.9.3 or newer is recommended.

1. Setup Legion for MPI interopability by installing GASNET. This is unfortunately not straightforward as GASNET setup is different for most systems and requires Legion specific setup. See the Legion GASNET instructions [here](http://legion.stanford.edu/gasnet/ "GASNET instructions").
2. Install Legion using these [instructions](http://legion.stanford.edu/starting/ "Legion Installation").
3. Install GeoPM following the instructions in the [repository](https://github.com/geopm/geopm "GeoPM").
4. Install [Libmsr](https://github.com/LLNL/libmsr "Libmsr"), which is used to verify GeoPM results. This requires either root or [msr_safe](https://github.com/LLNL/msr-safe "msr-safe").
5. Check the paths and configuration in ./tutorial_env.sh, you may need to change the job launcher or python paths for your system.
6. Change configuration in ./var.sh to reflect your local system. Depending on the job launcher used you may need to change the GEOPM_RM variable, see GeoPM documentation for details. LG_RT_DIR needs to point to xxx/legion/runtime. Change the GASNET variable to point to the install location of GASNET.
7. Change the variables in ./run.sh to reflect your local system. If you want to use a different GeoPM power configuration json file then you can set that here. GeoPM must be run in "process" mode.

## Run Directions
There are many nuances to building and executing a Legion+GeoPM program so I will show an example command and explain each part.

Before that you first need to "source ./var.sh" to get the environment variables necessary to compile and run your Legion+GeoPM program. Then you run "make", but you may want to do "make -j8" since each make builds the entire Legion runtime.

SLURM users must first allocate nodes (e.g. salloc) or GeoPM may not work correctly.

If you are not using SLURM or have a significantly different system, you will need to "./run.sh" and see if it works correctly. It likely will not so you need copy the command the run script generates (at the top) and modify it according to the explanations below.

Here is an example command that works on LLNL's 24 core SLURM systems:
~~~
GEOPM_TRACE=legion_daxpy_governed_trace GEOPM_PROFILE= GEOPM_POLICY=balanced_policy.json LD_DYNAMIC_WEAK=true GEOPM_REPORT=legion_daxpy_governed_report OMP_PROC_BIND=false GEOPM_PMPI_CTL=process MV2_ENABLE_AFFINITY=0 KMP_AFFINITY=disabled srun -N 2 -n 4 --mpibind=off --cpu_bind v,mask_cpu:0x2,0xfffffc ./daxpy_mpi 115.0 16 -ll:cpu 21 -ll:csize 8000
~~~

Step by step explanation:
* GEOPM_TRACE=legion_daxpy_governed_trace: the output trace file for GeoPM
* GEOPM_PROFILE= : the output profile for GeoPM
* GEOPM_POLICY=balanced_policy.json: the GeoPM power management policy to use
* LD_DYNAMIC_WEAK=true: Required by GeoPM
* GEOPM_REPORT=legion_daxpy_governed_report: the output report file for GeoPM
* GEOPM_PMPI_CTL=process: the mode to run GeoPM in, process mode is required for Legion interop
* OMP_PROC_BIND=false MV2_ENABLE_AFFINITY=0 KMP_AFFINITY=disabled: Legion is incompatible with any thread binding models, this ensures that they are all turned off
* srun -N 2 -n 4: each node has a Legion runtime as well as a GeoPM process.
* --mpibind=off: Legion is incompatible with mpibind so it must be disabled.
* --cpu_bind: GeoPM requires binding but mpibind's automatic binding interferes with Legion, so we must manually specify the binding of each MPI process with cpubind.
* v,mask_cpu:0x2,0xfffffc:  The first mask "0x2" is used for the first process (GeoPM), which we want to be bound to a single CPU. The second mask "0xfffffc" is for the second process (Legion) which we want to use cores 2-23. A single core "0x1" is left unbound for the OS which is recommended.
* ./daxpy_mpi 115.0 16: The daxpy program takes two arguments. The first will let you set a RAPL limit if GeoPM is disabled (via a macro). The second argument controls task granularity. You want this to be large enough to saturate the cores with tasks or higher.
-ll:cpu 21 -ll:csize 8000: The arguments beginning with "-ll" are commands for the Legion runtime. "-ll:cpu" is used to tell the runtime how many cores it can use. Note that this is one fewer than the cores given in the cpubind mask because one core is reserved for the Legion runtime, the others can execute Legion tasks. "-ll:csize" is used to tell the Legion runtime how much memory it will require, 8000 should work for the example unmodified.

## Comon Problems
	* reservation ('xxx') cannot be satisfied
		* the "-ll:cpu" argument is too large or too small
		* the cpubind mask is wrong
		* mpibind is not off
		* some thread binding model is enabled
	* Default mapper failed allocation for ...
		* the "-ll:csize" argument is too small
	* GASNET errors
		* you need to recompile and link GASNET with the correct settings
