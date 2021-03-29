# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol("N")
M = dace.symbol("M")


def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device")

    sdfg.add_array("A", [N, M], dtype=dace.float32)
    sdfg.add_array("C", [N, M], dtype=dace.float32)

    A_host = state.add_read("A")

    sdfg.add_array("A_device", [N, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("C_device", [N, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)

    A_device = state.add_write("A_device")

    state.add_edge(A_host, None, A_device, None, dace.Memlet("A_device"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to host

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_edge(C_device, None, C_host, None, dace.Memlet("C"))

    return state


def make_fpga_state(sdfg):

    state = sdfg.add_state("copy_m")

    A = state.add_read("A_device")

    C = state.add_write("C_device")


    ###########################################################################
    # A Read
    sdfg.add_stream('A_stream',
                    dtype=dace.float32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)


    read_a_tasklet = state.add_tasklet("read_a", {"mem"}, {"s"},
                    "s = mem")

    a_entry, a_exit = state.add_map("a_outer_map", {
        "n": "0:N",
        "m": "0:M"
        }, schedule=dace.ScheduleType.FPGA_Device)

    w_A_stream = state.add_write("A_stream")
    r_A_stream = state.add_read("A_stream")


    state.add_memlet_path(A,
                          a_entry,
                          read_a_tasklet,
                          dst_conn="mem",
                          memlet=dace.Memlet("A_device[n,m]"))

    state.add_memlet_path(read_a_tasklet,
                          a_exit,
                          w_A_stream,
                          src_conn="s",
                          memlet=dace.Memlet("A_stream[0]"))

    ###########################################################################
    # C Write
    sdfg.add_stream('C_stream',
                    dtype=dace.float32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    
    write_c_tasklet = state.add_tasklet("write_c", {"s"}, {"mem"}, "mem = s")

    c_entry, c_exit = state.add_map("c_outer_map", {
        "n": "0:N",
        "m": "0:M"
        }, schedule=dace.ScheduleType.FPGA_Device)

    w_C_stream = state.add_write("C_stream")
    r_C_stream = state.add_read("C_stream")

    state.add_memlet_path(r_C_stream,
                          c_entry,
                          write_c_tasklet,
                          dst_conn="s",
                          memlet=dace.Memlet("C_stream[0]"))

    state.add_memlet_path(write_c_tasklet,
                          c_exit,
                          C,
                          src_conn="mem",
                          memlet=dace.Memlet("C_device[n,m]"))

    ###########################################################################
    # Create and connect tasklet

    tasklet = state.add_tasklet(name="copy", inputs={"a_in"}, outputs={"c_out"}, code="c_out = a_in;")

    state.add_edge(r_A_stream, None, tasklet, "a_in", dace.Memlet("A_stream[0]"))
    state.add_edge(tasklet, "c_out", w_C_stream, None, dace.Memlet("C_stream[0]"))

    ###########################################################################

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("copy_matrix_{}x{}".format(
            N.get(), M.get()))
    else:
        sdfg = dace.SDFG("copy_matrix_Nx{}".format(M.get()))

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.InterstateEdge())

    return sdfg


if __name__ == "__main__":
    print("==== Program start ====")

    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    args = vars(parser.parse_args())

    M.set(args["M"])
    # M must always be specialized, as it's used for the static buffer size
    sdfg = make_sdfg(False)
    sdfg.specialize(dict(M=M))
    N.set(args["N"])

    print("Matrix copy {}x{}".format(
        M.get(), N.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    A[:] = np.random.rand(N.get(), M.get()).astype(dace.float32.type)
    C[:] = dace.float32(0)

    sdfg(A=A, C=C, N=N)

    diff = np.linalg.norm(A - C)
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")
