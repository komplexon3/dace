# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")


def make_rtl_taskled_sdfg():
    sdfg = dace.SDFG("nested_rtl")
    state = sdfg.add_state()

    sdfg.add_stream("rtl_A_stream",
                    dtype=dace.float32,
                    transient=False,
                    storage=dace.StorageType.FPGA_Local)
    r_A_stream = state.add_read("rtl_A_stream")

    sdfg.add_stream("rtl_B_stream",
                   dtype=dace.float32,
                   transient=False,
                   storage=dace.StorageType.FPGA_Local)
    r_B_stream = state.add_read("rtl_B_stream")

    sdfg.add_scalar("rtl_C_in_stream",
                   dtype=dace.float32,
                   transient=False,
                   storage=dace.StorageType.FPGA_Local)
    r_C_in_stream = state.add_read("rtl_C_in_stream")

    sdfg.add_scalar("rtl_C_out_stream",
                   dtype=dace.float32,
                   transient=False,
                   storage=dace.StorageType.FPGA_Local)
    w_C_out_stream = state.add_write("rtl_C_out_stream")

#    rtl_tasklet = state.add_tasklet(
#        name="rtl_ma",
#        inputs={"a", "b", "c_in"},
#        outputs={"c_out"},
#        code='''
#    assign ap_done = 1; // free-running
#    wire ap_aresetn = ~ap_areset;

#    wire        axis_ab_tvalid;
#    wire [31:0] axis_ab_tdata;
#    wire        axis_ab_tready;

#    floating_point_mult fl_mult (
#        .aclk(ap_aclk),
#        .aresetn(ap_aresetn),

#        .s_axis_a_tvalid(s_axis_a_tvalid),
#        .s_axis_a_tdata(s_axis_a_tdata),
#        .s_axis_a_tready(s_axis_a_tready),

#        .s_axis_b_tvalid(s_axis_b_tvalid),
#        .s_axis_b_tdata(s_axis_b_tdata),
#        .s_axis_b_tready(s_axis_b_tready),

#        .m_axis_result_tvalid(axis_ab_tvalid),
#        .m_axis_result_tdata(axis_ab_tdata),
#        .m_axis_result_tready(axis_ab_tready)
#    );

#    floating_point_add fl_add (
#        .aclk(ap_aclk),
#        .aresetn(ap_aresetn),

#        .s_axis_a_tvalid(s_axis_c_in_tvalid),
#        .s_axis_a_tdata(s_axis_c_in_tdata),
#        .s_axis_a_tready(s_axis_c_in_tready),

#        .s_axis_b_tvalid(axis_ab_tvalid),
#        .s_axis_b_tdata(axis_ab_tdata),
#        .s_axis_b_tready(axis_ab_tready),

#        .m_axis_result_tvalid(m_axis_c_out_tvalid),
#        .m_axis_result_tdata(m_axis_c_out_tdata),
#        .m_axis_result_tready(m_axis_c_out_tready)
#    );

#''', language=dace.Language.SystemVerilog)

#    rtl_tasklet.add_ip_core(
#    'floating_point_mult', 'floating_point', 'xilinx.com', '7.1', {
#        "CONFIG.Operation_Type": "Multiply",
#        "CONFIG.C_Mult_Usage": "Max_Usage",
#        "CONFIG.Axi_Optimize_Goal": "Performance",
#        "CONFIG.A_Precision_Type": "Single",
#        "CONFIG.C_A_Exponent_Width": "8",
#        "CONFIG.C_A_Fraction_Width": "24",
#        "CONFIG.Result_Precision_Type": "Single",
#        "CONFIG.C_Result_Exponent_Width": "8",
#        "CONFIG.C_Result_Fraction_Width": "24",
#        "CONFIG.C_Latency": "9",
#        "CONFIG.C_Rate": "1",
#        'CONFIG.Has_ARESETn': 'true',
#        "CONFIG.Flow_Control": "Blocking"
#    })

#    rtl_tasklet.add_ip_core(
#    'floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
#        "CONFIG.Add_Sub_Value": "Add",
#        "CONFIG.Axi_Optimize_Goal": "Performance",
#        'CONFIG.Has_ARESETn': 'true',
#        "CONFIG.Flow_Control": "Blocking"
#    })

    rtl_tasklet = state.add_tasklet(
        "multiply_accumulate", {"a", "b", "c_in"},
        {"c_out"}, """\
c_out = c_in + a * b""")

    # Connecting RTL Tasklet
    # In
    state.add_edge(r_A_stream, None, rtl_tasklet, "a", dace.Memlet("rtl_A_stream"))
    state.add_edge(r_B_stream, None, rtl_tasklet, "b", dace.Memlet("rtl_B_stream"))
    state.add_edge(r_C_in_stream, None, rtl_tasklet, "c_in", dace.Memlet("rtl_C_in_stream"))

    # Out
    state.add_edge(rtl_tasklet, "c_out", w_C_out_stream, None, dace.Memlet("rtl_C_out_stream"))

    return sdfg


def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device")

    sdfg.add_array("A", [N, K], dtype=dace.float32)
    sdfg.add_array("B", [K, M], dtype=dace.float32)
    sdfg.add_array("C", [N, M], dtype=dace.float32)

    A_host = state.add_read("A")
    B_host = state.add_read("B")
    C_host = state.add_read("C")

    sdfg.add_array("A_device", [N, K],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("C_device", [N, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)

    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")
    C_device = state.add_write("C_device")

    state.add_edge(A_host, None, A_device, None, dace.Memlet("A_device"))
    state.add_edge(B_host, None, B_device, None, dace.Memlet("B_device"))
    state.add_edge(C_host, None, C_device, None, dace.Memlet("C_device"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_edge(C_device, None, C_host, None, dace.Memlet("C"))

    return state


def make_fpga_state(sdfg):

    state = sdfg.add_state("mm")

    A = state.add_read("A_device")
    B = state.add_read("B_device")
    C = state.add_write("C_device")

    A_pipe_in = state.add_stream("A_pipe",
                                 dace.float32,
                                 transient=True,
                                 storage=dace.StorageType.FPGA_Local)
    B_pipe_in = state.add_stream("B_pipe",
                                 dace.float32,
                                 transient=True,
                                 storage=dace.StorageType.FPGA_Local)
    C_pipe_in = state.add_stream("C_pipe",
                                 dace.float32,
                                 transient=True,
                                 storage=dace.StorageType.FPGA_Local)
    A_pipe_out = state.add_stream("A_pipe",
                                  dace.float32,
                                  transient=True,
                                  storage=dace.StorageType.FPGA_Local)
    B_pipe_out = state.add_stream("B_pipe",
                                  dace.float32,
                                  transient=True,
                                  storage=dace.StorageType.FPGA_Local)
    C_pipe_out = state.add_stream("C_pipe",
                                  dace.float32,
                                  transient=True,
                                  storage=dace.StorageType.FPGA_Local)

    state.add_memlet_path(A, A_pipe_out, memlet=dace.Memlet("A_device"))

    read_b_entry, read_b_exit = state.add_map(
        "read_b", {
            "n": "0:N",
            "m": "0:M",
            "k": "0:K"
        },
        schedule=dace.ScheduleType.FPGA_Device)
    read_b_tasklet = state.add_tasklet("read_b", {"mem"}, {"s"}, "s = mem")
    state.add_memlet_path(B,
                          read_b_entry,
                          read_b_tasklet,
                          dst_conn="mem",
                          memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(read_b_tasklet,
                          read_b_exit,
                          B_pipe_out,
                          src_conn="s",
                          memlet=dace.Memlet("B_pipe[0]"))

    state.add_memlet_path(C_pipe_in,
                          C,
                          src_conn="mem",
                          memlet=dace.Memlet("C_device"))

    ###########################################################################

    n_entry, n_exit = state.add_map("outer_map", {"n": "0:N"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    km_entry, km_exit = state.add_map("inner_map", {
        "m": "0:M",
        "k": "0:K"
    },
                                      schedule=dace.ScheduleType.FPGA_Device)

    sdfg.add_array("output_buffer", [M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    output_buffer_read = state.add_read("output_buffer")
    output_buffer_write = state.add_write("output_buffer")

    nested_tasklet = state.add_nested_sdfg(make_rtl_taskled_sdfg(), None, {'rtl_A_stream', 'rtl_B_stream', 'rtl_C_in_stream'}, {'rtl_C_out_stream'})

    state.add_memlet_path(A_pipe_in,
                          n_entry,
                          km_entry,
                          nested_tasklet,
                          dst_conn="rtl_A_stream",
                          memlet=dace.Memlet("A_pipe[0]", dynamic=True))

    state.add_memlet_path(B_pipe_in,
                          n_entry,
                          km_entry,
                          nested_tasklet,
                          dst_conn="rtl_B_stream",
                          memlet=dace.Memlet("B_pipe[0]"))

    state.add_memlet_path(output_buffer_read,
                          km_entry,
                          nested_tasklet,
                          dst_conn="rtl_C_in_stream",
                          memlet=dace.Memlet("output_buffer[m]"))

    # Make sure it's in scope
    state.add_memlet_path(n_entry, output_buffer_read, memlet=dace.Memlet())

    state.add_memlet_path(nested_tasklet,
                          km_exit,
                          output_buffer_write,
                          src_conn="rtl_C_out_stream",
                          memlet=dace.Memlet("output_buffer[m]"))

    state.add_memlet_path(output_buffer_write,
                          n_exit,
                          C_pipe_out,
                          memlet=dace.Memlet("output_buffer[0:M]"))

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("rtl_mm_nested_{}x{}x{}".format(
            N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("rtl_mm_nested_NxKx{}".format(M.get()))

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
    parser.add_argument("K", type=int)
    parser.add_argument("-specialize",
                        default=False,
                        action="store_true",
                        help="Fix all loop bounds at compile time/in hardware")
    args = vars(parser.parse_args())

    if not args["specialize"]:
        M.set(args["M"])
        # M must always be specialized, as it's used for the static buffer size
        sdfg = make_sdfg(False)
        sdfg.specialize(dict(M=M))
        N.set(args["N"])
        K.set(args["K"])
    else:
        M.set(args["M"])
        N.set(args["N"])
        K.set(args["K"])
        sdfg = make_sdfg(True)
        sdfg.specialize(dict(M=M, N=N, K=K))

    print("Matrix multiplication {}x{}x{} ({}specialized)".format(
        M.get(), N.get(), K.get(), "" if args["specialize"] else "not "))

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get(), K.get()], dtype=dace.float32.type)
    B = np.ndarray([K.get(), M.get()], dtype=dace.float32.type)
    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    A[:] = 1  # np.random.rand(N.get(), K.get()).astype(dace.float32.type)
    B[:] = 1  # np.random.rand(K.get(), M.get()).astype(dace.float32.type)
    C[:] = dace.float32(0)

    A_regression = np.ndarray([N.get(), K.get()], dtype=np.float32)
    B_regression = np.ndarray([K.get(), M.get()], dtype=np.float32)
    C_regression = np.ndarray([N.get(), M.get()], dtype=np.float32)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    if args["specialize"]:
        sdfg(A=A, B=B, C=C)
    else:
        sdfg(A=A, B=B, C=C, N=N, K=K)

    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")
