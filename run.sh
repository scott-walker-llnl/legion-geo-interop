source tutorial_env.sh
GEOPM_RM=SrunLauncher
GEOPMPY_PKGDIR=$LOCAL/lib/python2.7/site-packages/geopmpy
geopmsrun	-N2 -n2 \
			--geopm-ctl=process \
			--geopm-report=legion_daxpy_governed_report \
			--geopm-trace=legion_daxpy_governed_trace \
			--geopm-policy=balanced_policy.json \
			-- ./daxpy_mpi 115.0 8
