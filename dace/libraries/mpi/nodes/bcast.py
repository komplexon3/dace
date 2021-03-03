# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace import data as dt, dtypes, memlet as mm, SDFG, SDFGState, symbolic
from dace.frontend.common import op_repository as oprepo


@dace.library.expansion
class ExpandBcastPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of MPI Bcast.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise(NotImplementedError)
  
@dace.library.expansion
class ExpandBcastMPI(ExpandTransformation):

    environments = [environments.mpi.MPI]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (buffer, count_str), root = node.validate(
            parent_sdfg, parent_state)
        dtype = buffer.dtype.base_type
        mpi_dtype_str = "MPI_BYTE"
        if dtype == dace.dtypes.float32:
            mpi_dtype_str = "MPI_FLOAT"
        elif dtype == dace.dtypes.float64:
            mpi_dtype_str = "MPI_DOUBLE" 
        elif dtype == dace.dtypes.complex64:
            mpi_dtype_str = "MPI_COMPLEX"
        elif dtype == dace.dtypes.complex128:
            mpi_dtype_str = "MPI_COMPLEX_DOUBLE"
        elif dtype == dace.dtypes.int32:
            mpi_dtype_str = "MPI_INT"

        else:
            print("The datatype "+str(dtype)+" is not supported!")
            raise(NotImplementedError) 
        if buffer.dtype.veclen > 1:
            raise(NotImplementedError)
        if root.dtype.base_type != dace.dtypes.int32:
            raise ValueError("Bcast root must be an integer!")

        code = f"MPI_Bcast(_buffer, {count_str}, {mpi_dtype_str}, _root, MPI_COMM_WORLD);"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.node
class Bcast(dace.sdfg.nodes.LibraryNode):

    # Global properties
    implementations = {
        "MPI": ExpandBcastMPI,
    }
    default_implementation = "MPI"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"_buffer", "_root"},
                         outputs={"_buffer"},
                         **kwargs)

    def validate(self, sdfg, state):
        """
        :return: A three-tuple (buffer, root) of the three data descriptors in the
                 parent SDFG.
        """
        
        inbuffer, outbuffer, src, tag = None, None, None, None
        for e in state.out_edges(self):
            if e.src_conn == "_buffer":
                outbuffer = sdfg.arrays[e.data.data]
        for e in state.in_edges(self):
            if e.dst_conn == "_buffer":
                inbuffer = sdfg.arrays[e.data.data]
            if e.dst_conn == "_root":
                root = sdfg.arrays[e.data.data]

        if inbuffer != outbuffer:
            raise(ValueError("Bcast input and output buffer must be the same!"))
        if root.dtype.base_type != dace.dtypes.int32:
            raise(ValueError("Bcast root must be an integer!"))

        count_str = "XXX"
        for _, src_conn, _, _, data in state.out_edges(self):  
            if src_conn == '_buffer':
                dims = [str(e) for e in data.subset.size_exact()]
                count_str = "*".join(dims)

        return (inbuffer, count_str), root

