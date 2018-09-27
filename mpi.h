// this is used for checking Legion+MPI+GeoPM interactions
#include <mpi.h>
#include <cstdio>

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */

#ifdef MPICH_HAS_C2F
_EXTERN_C_ void *MPIR_ToPointer(int);
#endif // MPICH_HAS_C2F

#ifdef PIC
/* For shared libraries, declare these weak and figure out which one was linked
   based on which init wrapper was called.  See mpi_init wrappers.  */
#pragma weak pmpi_init
#pragma weak PMPI_INIT
#pragma weak pmpi_init_
#pragma weak pmpi_init__
#endif /* PIC */

_EXTERN_C_ void pmpi_init(MPI_Fint *ierr);
_EXTERN_C_ void PMPI_INIT(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init_(MPI_Fint *ierr);
_EXTERN_C_ void pmpi_init__(MPI_Fint *ierr);

_EXTERN_C_ int PMPI_Init(int *argc, char ***argv);
_EXTERN_C_ int MPI_Init(int *argc, char ***argv) {
    int _wrap_py_return_val = 0;

	printf("INTERCEPTED PRE INIT\n");
    _wrap_py_return_val = PMPI_Init(argc, argv);
	printf("INTERCEPTED INIT\n");

    return _wrap_py_return_val;
}

_EXTERN_C_ int PMPI_Init_thread(int *argc, char ***argv, int required, int *provided);
_EXTERN_C_ int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int _wrap_py_return_val = 0;

	printf("INTERCEPTED PRE THREAD INIT\n");
    _wrap_py_return_val = PMPI_Init_thread(argc, argv, required, provided);
	// printf("INTERCEPTED: provided %d\n", *provided);
	// *provided = 3;
	printf("INTERCEPTED POST THREAD INIT\n");

    return _wrap_py_return_val;
}


_EXTERN_C_ int PMPI_Finalize();
_EXTERN_C_ int MPI_Finalize() {
    int _wrap_py_return_val = 0;

    _wrap_py_return_val = PMPI_Finalize();
	printf("INTERCEPTED FINALIZE\n");
    return _wrap_py_return_val;
}
