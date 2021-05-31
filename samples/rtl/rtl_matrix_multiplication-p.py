# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#
#                    s_axis_b
#                       │
#             s_axis_a  │
#                │      │
#   ap_aclk      │      │
#      │         │      │
#      │     ┌───▼──────▼───┐
#      │     │              │
#      ├────►│  Multiplier  │
#      │     │              │
#      │     └───────┬──────┘
#      │             │
#      │             │  s_axis_c_in
#      │             │     │
#      │         ┌───▼─────▼───┐
#      │         │             │
#      └────────►│   Adder     │
#                │             │
#                └─────┬───────┘
#                      │
#                      │
#                      │
#                      ▼
#                 m_axis_c_out

import argparse
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")


def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device", is_start_state=True)

    sdfg.add_array("A", [N, K], dtype=dace.float32)
    sdfg.add_array("B", [K, M], dtype=dace.float32)
    sdfg.add_array("C", [N, M], dtype=dace.float32)

    A_host = state.add_read("A")
    B_host = state.add_read("B")

    sdfg.add_array("A_device", [N, K],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)

    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")

    state.add_edge(A_host, None, A_device, None, dace.Memlet("A_device"))
    state.add_edge(B_host, None, B_device, None, dace.Memlet("B_device"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to host

    state = sdfg.add_state("copy_to_host")

    Output_buffer = state.add_read("output_buffer")
    C_host = state.add_write("C")

    state.add_edge(Output_buffer, None, C_host, None, dace.Memlet("C"))

    return state

def make_fpga_state(sdfg, factor):

    state = sdfg.add_state("rtl_mm")

    A = state.add_read("A_device")
    B = state.add_read("B_device")

    sdfg.add_array("output_buffer", [N,M],
                dtype=dace.float32,
                transient=True,
                storage=dace.StorageType.FPGA_Global)
    r_output_buffer = state.add_read("output_buffer")
    w_output_buffer = state.add_write("output_buffer")


    for p in range(factor):
        n_range = ""
        if p == factor-1:
            n_range = f"({p}*(N//{factor})):N"
        else:
            n_range = f"({p}*(N//{factor})):(({p}+1)*(N//{factor}))"
        print(n_range)

        sdfg.add_stream(f"A_stream_{p}",
                        dtype=dace.float32,
                        transient=True,
                        storage=dace.StorageType.FPGA_Local)
        r_A_stream = state.add_read(f"A_stream_{p}")
        w_A_stream = state.add_write(f"A_stream_{p}")

        sdfg.add_stream(f"B_stream_{p}",
                    dtype=dace.float32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
        r_B_stream = state.add_read(f"B_stream_{p}")
        w_B_stream = state.add_read(f"B_stream_{p}")

        sdfg.add_stream(f"output_stream_{p}",
                    dtype=dace.float32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
        r_output_stream = state.add_read(f"output_stream_{p}")
        w_output_stream = state.add_write(f"output_stream_{p}")

        sdfg.add_stream(f"c_out_stream_{p}",
                    dtype=dace.float32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
        r_c_out_stream = state.add_read(f"c_out_stream_{p}")
        w_c_out_stream = state.add_write(f"c_out_stream_{p}")

        ###########################################################################
        ### Process input 
        # In a map
        in_entry, in_exit = state.add_map(
            f"in_map_{p}",
            dict(k="0:K", n=n_range, m="0:M"),
            schedule=dace.ScheduleType.FPGA_Device)

        # Input a processing tasklet
        read_in = state.add_tasklet(f"read_in_{p}",
                                    {"a_in"},
                                    {"a_out"},
                                    """
a_out = a_in
                                """)

        state.add_memlet_path(A,
                            in_entry,
                            read_in,
                            dst_conn="a_in",
                            memlet=dace.Memlet("A_device[n, k]"))
        state.add_memlet_path(read_in,
                            in_exit,
                            w_A_stream,
                            src_conn="a_out",
                            memlet=dace.Memlet(f"A_stream_{p}[0]"))

        # In b map
        in_b_entry, in_b_exit = state.add_map(
            f"in_b_map_{p}",
            dict(k="0:K", n=n_range, m="0:M"),
            schedule=dace.ScheduleType.FPGA_Device)

        # Input processing tasklet
        read_b_in = state.add_tasklet(f"read_b_in_{p}",
                                    {"b_in"},
                                    {"b_out"},
                                    """
b_out = b_in
                                    """)

        state.add_memlet_path(B,
                            in_b_entry,
                            read_b_in,
                            dst_conn="b_in",
                            memlet=dace.Memlet("B_device[k, m]"))
        state.add_memlet_path(read_b_in,
                            in_b_exit,
                            w_B_stream,
                            src_conn="b_out",
                            memlet=dace.Memlet(f"B_stream_{p}[0]"))

        # In output buffer map
        in_buff_entry, in_buff_exit = state.add_map(
            f"in_buff_map_{p}",
            dict(k="0:K", n=n_range, m="0:M"),
            schedule=dace.ScheduleType.FPGA_Device)

        # Input processing tasklet
        read_buff_in = state.add_tasklet(f"read_buff_in_{p}",
                                    {"out_in"},
                                    {"out_out"},
                                    """
out_out=out_in
                                    """)

        state.add_memlet_path(r_output_buffer,
                            in_buff_entry,
                            read_buff_in,
                            dst_conn="out_in",
                            memlet=dace.Memlet("output_buffer[n,m]"))
        state.add_memlet_path(read_buff_in,
                            in_buff_exit,
                            w_output_stream,
                            src_conn="out_out",
                            memlet=dace.Memlet(f"output_stream_{p}[0]"))

        ###########################################################################
        ### Process c_out -> output buffer 
        # Map
        out_entry, out_exit = state.add_map(
            f"out_map{p}",
            dict(k="0:K", n=n_range, m="0:M"),
            schedule=dace.ScheduleType.FPGA_Device)

        proc_out = state.add_tasklet("proc_out", {"c_out"}, {"out_buf"}, "out_buf = c_out")

        state.add_memlet_path(r_c_out_stream,
                            out_entry,
                            proc_out,
                            dst_conn="c_out",
                            memlet=dace.Memlet(f"c_out_stream_{p}[0]"))
        state.add_memlet_path(proc_out,
                            out_exit,
                            w_output_buffer,
                            src_conn="out_buf",
                            memlet=dace.Memlet("output_buffer[n,m]"))

        ###########################################################################
        # Multiply accumulate RTL tasklet

        #base_clk_freq = dace.Config.get('compiler', 'xilinx', 'frequency')
        #double_clk_freq = 2 * base_clk_freq

        rtl_tasklet = state.add_tasklet(
            name=f"rtl_ma_{p}",
            inputs={"a", "b", "c_in"},
            outputs={"c_out"},
            code='''
        assign ap_done = 1; // free-running
        wire ap_aresetn = ~ap_areset;

        wire        axis_ab_tvalid;
        wire [31:0] axis_ab_tdata;
        wire        axis_ab_tready;

        floating_point_mult fl_mult (
            .aclk(ap_aclk),
            .aresetn(ap_aresetn),

            .s_axis_a_tvalid(s_axis_a_tvalid),
            .s_axis_a_tdata(s_axis_a_tdata),
            .s_axis_a_tready(s_axis_a_tready),

            .s_axis_b_tvalid(s_axis_b_tvalid),
            .s_axis_b_tdata(s_axis_b_tdata),
            .s_axis_b_tready(s_axis_b_tready),

            .m_axis_result_tvalid(axis_ab_tvalid),
            .m_axis_result_tdata(axis_ab_tdata),
            .m_axis_result_tready(axis_ab_tready)
        );

        floating_point_add fl_add (
            .aclk(ap_aclk),
            .aresetn(ap_aresetn),

            .s_axis_a_tvalid(s_axis_c_in_tvalid),
            .s_axis_a_tdata(s_axis_c_in_tdata),
            .s_axis_a_tready(s_axis_c_in_tready),

            .s_axis_b_tvalid(axis_ab_tvalid),
            .s_axis_b_tdata(axis_ab_tdata),
            .s_axis_b_tready(axis_ab_tready),

            .m_axis_result_tvalid(m_axis_c_out_tvalid),
            .m_axis_result_tdata(m_axis_c_out_tdata),
            .m_axis_result_tready(m_axis_c_out_tready)
        );

    ''', language=dace.Language.SystemVerilog)

        rtl_tasklet.add_ip_core(
        'floating_point_mult', 'floating_point', 'xilinx.com', '7.1', {
            "CONFIG.Operation_Type": "Multiply",
            "CONFIG.C_Mult_Usage": "Max_Usage",
            "CONFIG.Axi_Optimize_Goal": "Performance",
            "CONFIG.A_Precision_Type": "Single",
            "CONFIG.C_A_Exponent_Width": "8",
            "CONFIG.C_A_Fraction_Width": "24",
            "CONFIG.Result_Precision_Type": "Single",
            "CONFIG.C_Result_Exponent_Width": "8",
            "CONFIG.C_Result_Fraction_Width": "24",
            "CONFIG.C_Latency": "9",
            "CONFIG.C_Rate": "1",
            'CONFIG.Has_ARESETn': 'true',
            "CONFIG.Flow_Control": "Blocking"
        })

        rtl_tasklet.add_ip_core(
        'floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
            "CONFIG.Add_Sub_Value": "Add",
            "CONFIG.Axi_Optimize_Goal": "Performance",
            'CONFIG.Has_ARESETn': 'true',
            "CONFIG.Flow_Control": "Blocking"
        })

        # Connecting RTL Tasklet
        # In
        state.add_edge(r_A_stream, None, rtl_tasklet, "a", dace.Memlet(f"A_stream_{p}[0]"))
        state.add_edge(r_B_stream, None, rtl_tasklet, "b", dace.Memlet(f"B_stream_{p}[0]"))
        state.add_edge(r_output_stream, None, rtl_tasklet, "c_in", dace.Memlet(f"output_stream_{p}[0]"))

        # Out
        state.add_edge(rtl_tasklet, "c_out", w_c_out_stream, None, dace.Memlet(f"c_out_stream_{p}[0]"))

    return state


def make_sdfg(p):

    sdfg = dace.SDFG("rtl_matrix_multiplication_p")

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg, p)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.InterstateEdge())

    return sdfg


def probe_max_freq():
    from dace.codegen import exceptions as cgx
    
    min = 462
    max = 500

    print(f"=== Building SDFG - {min} to {max} @ 10 ===")
    sdfg = make_sdfg(10)

    print(f"=== Start Probing - {min} to {max} @ 10 ===")

    while max-min > 1:
        print(f"### Current range: {min} to {max} @ 10 ###")
        testing = (min+max)//2
        dace.Config.set('compiler', 'xilinx', 'frequency', value=str(testing))
        print (f"~~~~ Test Frequency: {testing} @ 10 ~~~")
        try:
            sdfg.compile()
            min = testing
            print (f"+++ Success for: {testing} @ 10 +++")
        except cgx.CompilationError: 
            max = testing
            print (f"--- Failure for: {testing} @ 10 ---")
        except: 
            print (f"!!! Unexpected error for: {testing} @ 10 !!!")
            print(f"Error:", sys.exc_info()[0])
            raise

if __name__ == "__main__":
    print("==== Probing max freq ====")
    probe_max_freq()

    sdfg = make_sdfg()
    testing = 1000 #(min+max)//2
    dace.Config.set('compiler', 'xilinx', 'frequency', value=str(testing))
    print (f"~~~~ Test Frequency: {testing} ~~~")
    print(dace.Config.get('compiler', 'xilinx', 'frequency'))
    sdfg.compile()
    print(dace.Config.get('compiler', 'xilinx', 'frequency'))

#    print("==== Program start ====")

#    parser = argparse.ArgumentParser()
#    parser.add_argument("M", type=int)
#    parser.add_argument("N", type=int)
#    parser.add_argument("K", type=int)
#    parser.add_argument("P", type=int)
#    args = vars(parser.parse_args())

#    # unroll factor
#    p = args["P"]

#    M.set(args["M"])
#    N.set(args["N"])
#    K.set(args["K"])
#    sdfg = make_sdfg(p)

#    print("Matrix multiplication {}x{}x{}".format(
  #      M.get(), N.get(), K.get()))

#    # Initialize arrays: Randomize A and B, zero C
#    A = np.ndarray([N.get(), K.get()], dtype=dace.float32.type)
#    B = np.ndarray([K.get(), M.get()], dtype=dace.float32.type)
#    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
##    A[:] = [
##        [dace.float32(1),dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(0)],
##        [dace.float32(0),dace.float32(1),dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(0)],
##        [dace.float32(0),dace.float32(0),dace.float32(1),dace.float32(0),dace.float32(0),dace.float32(0)],
##        [dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(1),dace.float32(0),dace.float32(0)],
##        [dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(1),dace.float32(0)],
##        [dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(0),dace.float32(1)]
##    ]
#    A[:] = np.random.rand(N.get(), K.get()).astype(dace.float32.type)
##    B[:] = [
##        [dace.float32(1), dace.float32(2), dace.float32(3), dace.float32(4), dace.float32(5), dace.float32(6)],
##        [dace.float32(7), dace.float32(8), dace.float32(9), dace.float32(10), dace.float32(11), dace.float32(12)],
##        [dace.float32(13), dace.float32(14), dace.float32(15), dace.float32(16), dace.float32(17), dace.float32(18)],
##        [dace.float32(19), dace.float32(20), dace.float32(21), dace.float32(22), dace.float32(23), dace.float32(24)],
##        [dace.float32(25), dace.float32(26), dace.float32(27), dace.float32(28), dace.float32(29), dace.float32(30)],
##        [dace.float32(31), dace.float32(32), dace.float32(33), dace.float32(34), dace.float32(35), dace.float32(36)]
##    ]
#    B[:] = np.random.rand(K.get(), M.get()).astype(dace.float32.type)

#    A_regression = np.ndarray([N.get(), K.get()], dtype=np.float32)
#    B_regression = np.ndarray([K.get(), M.get()], dtype=np.float32)
#    C_regression = np.ndarray([N.get(), M.get()], dtype=np.float32)
#    A_regression[:] = A[:]
#    B_regression[:] = B[:]
#    C_regression[:] = C[:]

#    sdfg(A=A, B=B, C=C, M=M, N=N, K=K)

##    print("---------------------")
##    print(A)
##    print("*********************")
##    print(B)
##    print("=====================")
##    print(C)
##    print("---------------------")


#    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
#    if diff > 1e-6:
  #      raise ValueError(f"Verification failed, difference: {diff}")
#    else:
  #      print("Results successfully verified.")