# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
from dace.codegen.exceptions import CompilerConfigurationError, CompilationError
import dace.libraries.mpi as mpi
import sys
import warnings
import numpy as np
from mpi4py import MPI as MPI4PY


###############################################################################


def make_sdfg(dtype):

    n = dace.symbol("n")

    sdfg = dace.SDFG("mpi_send_recv")
    state = sdfg.add_state("dataflow")

    sdfg.add_array("x", [n], dtype, transient=False)
    sdfg.add_array("root", [1], dace.dtypes.int32, transient=False)
    x = state.add_access("x")
    xout = state.add_access("x")
    root = state.add_access("root")
    bcast_node = mpi.nodes.bcast.Bcast("bcast")
    
    state.add_memlet_path(x,
                          bcast_node,
                          dst_conn="_buffer",
                          memlet=Memlet.simple(x, "0:n", num_accesses=n))
    state.add_memlet_path(root,
                          bcast_node,
                          dst_conn="_root",
                          memlet=Memlet.simple(root, "0:1", num_accesses=1))
    state.add_memlet_path(bcast_node,
                          xout,
                          src_conn="_buffer",
                          memlet=Memlet.simple(xout, "0:n", num_accesses=1))

    return sdfg


###############################################################################


def _test_mpi(info, sdfg, dtype):
    comm = MPI4PY.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()
    mpi_sdfg = None
    if commsize < 2:
        raise ValueError("This test is supposed to be run with at least two processes!")
    for r in range(0, commsize):
        if r == rank:
            mpi_sdfg = sdfg.compile()
        comm.Barrier()
  
    size = 128
    A = np.full(size, rank, dtype=dtype)
    root = np.array([23], dtype=np.int32)
    mpi_sdfg(x=A,root=root, n=size)
    # now B should be an array of size, containing 0
    if not np.allclose(B, np.full(size, 0, dtype=dtype)):
        raise(ValueError("The received values are not what I expected."))

def test_mpi():
    _test_mpi("MPI Send/Recv", make_sdfg(np.float), np.float)

###############################################################################

if __name__ == "__main__":
    test_mpi()
###############################################################################
