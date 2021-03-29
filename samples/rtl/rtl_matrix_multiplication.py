# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    # Copy data to host

    state = sdfg.add_state("copy_to_host")

    C_device = state.add_read("C_device")
    C_host = state.add_write("C")

    state.add_edge(C_device, None, C_host, None, dace.Memlet("C"))

    return state

def make_fpga_state(sdfg):

    state = sdfg.add_state("rtl_mm")

    A = state.add_read("A_device")
    B = state.add_read("B_device")
    C = state.add_write("C_device")

    sdfg.add_stream("A_stream",
                   dace.float32,
                    transient=True,
                    storage=dace.StorageType.FPGA_Local)
    r_A_stream = state.add_read("A_stream")
    w_A_stream = state.add_write("A_stream")

    sdfg.add_stream("B_stream",
                   dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    r_B_stream = state.add_read("B_stream")
    w_B_stream = state.add_read("B_stream")

    sdfg.add_stream("C_stream",
                   dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    r_C_stream = state.add_read("C_stream")
    w_C_stream = state.add_read("C_stream")

    sdfg.add_array("output_buffer", [M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    r_output_buffer = state.add_read("output_buffer")
    #w_output_buffer = state.add_write("output_buffer")
    #ro_output_buffer = state.add_read("output_buffer")
    wo_output_buffer = state.add_write("output_buffer")

    sdfg.add_stream("output_stream",
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    r_output_stream = state.add_read("output_stream")
    w_output_stream = state.add_write("output_stream")

    sdfg.add_stream("c_out_stream",
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    r_c_out_stream = state.add_read("c_out_stream")
    w_c_out_stream = state.add_write("c_out_stream")

    sdfg.add_stream("a_buff_stream",
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    r_a_buff_stream = state.add_read("a_buff_stream")
    w_a_buff_stream = state.add_write("a_buff_stream")


    ###########################################################################
    ### Process input 
    # In map
    in_n_entry, in_n_exit = state.add_map("in_outer_map", {"n": "0:N"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    in_km_entry, in_km_exit = state.add_map("in_inner_map", {
        "k": "0:K",
        "m": "0:M"
    },
                                    schedule=dace.ScheduleType.FPGA_Device)

    # Input processing tasklet
    read_in = state.add_tasklet("read_in",
                                {"a_in", "a_reg", "b_in", "out_in"},
                                {"a_out", "b_out", "out_out"},
                                """
a_out = a_in if m == 0 else a_reg
b_out = b_in
out_out = 0 if k == 0 else out_in
out_out = out_in
                                """)

    # A[n,k] -> A_stream processing path
    # a buffer stream (a_reg) -> A_stream processing path
    # multiplexed
    state.add_memlet_path(r_a_buff_stream,
                          in_n_entry,
                          in_km_entry,
                          read_in,
                          dst_conn="a_reg",
                          memlet=dace.Memlet("a_buff_stream[0]"))
    state.add_memlet_path(A,
                          in_n_entry,
                          in_km_entry,
                          read_in,
                          dst_conn="a_in",
                          memlet=dace.Memlet("A_device[n, k]"))
    state.add_memlet_path(read_in,
                          in_km_exit,
                          in_n_exit,
                          w_A_stream,
                          src_conn="a_out",
                          memlet=dace.Memlet("A_stream[0]"))

    # B[k,m] -> B_stream processing path
    state.add_memlet_path(B,
                          in_n_entry,
                          in_km_entry,
                          read_in,
                          dst_conn="b_in",
                          memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(read_in,
                          in_km_exit,
                          in_n_exit,
                          w_B_stream,
                          src_conn="b_out",
                          memlet=dace.Memlet("B_stream[0]"))
    
    # output_buffer[m] -> output_reg processing path
    state.add_memlet_path(r_output_buffer,
                          in_n_entry,
                          in_km_entry,
                          read_in,
                          dst_conn="out_in",
                          memlet=dace.Memlet("output_buffer[m]"))
    state.add_memlet_path(read_in,
                          in_km_exit,
                          in_n_exit,
                          w_output_stream,
                          src_conn="out_out",
                          memlet=dace.Memlet("output_stream[0]"))

    ###########################################################################
    ### Process c_out -> output buffer 
    # Map
    out_n_entry, out_n_exit = state.add_map("out_outer_map", {"n": "0:N"},
                                    schedule=dace.ScheduleType.FPGA_Device)
    out_km_entry, out_km_exit = state.add_map("out_inner_map", {
        "k": "0:K",
        "m": "0:M"
    },
                                    schedule=dace.ScheduleType.FPGA_Device)

    # Output (c_out -> output_buffer) processing tasklet
    proc_out = state.add_tasklet("proc_out", {"c_out"}, {"out_buf"}, "c_out = out_buf")

    # c_out -> output buffer processing path
    state.add_memlet_path(r_c_out_stream,
                          out_n_entry,
                          out_km_entry,
                          proc_out,
                          dst_conn="c_out",
                          memlet=dace.Memlet("c_out_stream[0]"))
    state.add_memlet_path(proc_out,
                          out_km_exit,
                          wo_output_buffer,
                          src_conn="out_buf",
                          memlet=dace.Memlet("output_buffer[m]"))
    # Make sure output_buffer is in scope
    # state.add_memlet_path(out_n_entry, out_km_entry, out_km_exit, wo_output_buffer, memlet=dace.Memlet())
    state.add_memlet_path(wo_output_buffer,
                          out_n_exit,
                          w_C_stream,
                          memlet=dace.Memlet("output_buffer[0:M]"))

    ###########################################################################
    # Writing to C_device
    state.add_memlet_path(r_C_stream,
                          C,
                          src_conn="mem",
                          memlet=dace.Memlet("C_device"))

    ###########################################################################
    # Multiply accumulate RTL tasklet

    #base_clk_freq = dace.Config.get('compiler', 'xilinx', 'frequency')
    #double_clk_freq = 2 * base_clk_freq

    rtl_tasklet = state.add_tasklet(
        name="rtl_multiply_accumulate",
        inputs={"a", "b", "c_in"},
        outputs={"a_reg_out", "c_out"},
        code='''
       
    assign ap_done = 1; // free-running

    wire [31:0] a_buffer_in;
    wire [31:0] prev_buffer_in;
    wire a_valid_in;
    wire prev_valid_in;
    reg [31:0] a_buffer_out;
    reg [31:0] prev_buffer_out;
    reg a_valid_out;
    reg prev_valid_out;

    // drop these unnecessary ass and use signals directly
    // ready if a pipe input valid or if we don't need to consume it
    assign a_valid_in = s_axis_a_tvalid;
    assign prev_valid_in = s_axis_c_in_tvalid;

    // multiplexer a <= m == 0 ? s_axis_a_mem_tdata : a_reg_in -> computed in input map
    // should be buffered for 9 + 14 = 21 cycles
    assign a_buffer_in = s_axis_a_tdata;


    // multiplexer prev <= k == 0 ? 0 : output_buffer[m] -> computed in input map
    // should be buffered for 9 cycles
    assign prev_buffer_in = s_axis_c_in_tdata;
    
    // ===================================
    // inline delay buffers for a and prev
    localparam integer DELAY_A=14;
    localparam integer DELAY_PREV=9;
    
    reg [31:0] delay_a_mem[DELAY_A-1:0];
    reg [31:0] delay_prev_mem[DELAY_PREV-1:0];
    reg delay_a_valid_mem[DELAY_A-1:0];
    reg delay_prev_valid_mem[DELAY_PREV-1:0];
    
    reg [$clog2(DELAY_A)-1:0] ptr_a;
    reg [$clog2(DELAY_PREV)-1:0] ptr_prev;

    wire        axis_ab_tvalid;
    wire [31:0] axis_ab_tdata;
    wire        axis_ab_tready;
    
    always @(posedge ap_aclk) begin
        if (!ap_areset)
            ptr_a <= 0;
        else if (ptr_a+1 == DELAY_A)
	      ptr_a <= 0;
      else
	      ptr_a <= ptr_a+1;
	end
	
	always @(posedge ap_aclk) begin
        if (!ap_areset)
            ptr_prev <= 0;
        else if (ptr_prev+1 == DELAY_A)
	      ptr_prev <= 0;
      else
	      ptr_prev <= ptr_prev+1;
	end
	
	always @(posedge ap_aclk) begin
	   delay_a_mem[ptr_a] <= a_buffer_in;
	   a_buffer_out <= delay_a_mem[ptr_a];

	   delay_prev_mem[ptr_prev] <= prev_buffer_in;
	   prev_buffer_out <= delay_prev_mem[ptr_prev];

	   delay_a_valid_mem[ptr_a] <= a_valid_in;
	   a_valid_out <= delay_a_valid_mem[ptr_a];

	   delay_prev_valid_mem[ptr_prev] <= prev_valid_in;
	   prev_valid_out <= delay_prev_valid_mem[ptr_prev];
	end
    // ===================================

    // latency 9
    floating_point_mult fl_mult (
        .aclk(ap_aclk),

        .s_axis_a_tvalid(a_valid_in),
        .s_axis_a_tdata(a_buffer_in),
        .s_axis_a_tready(s_axis_a_tready),

        .s_axis_b_tvalid(s_axis_b_tvalid),
        .s_axis_b_tdata(s_axis_b_tdata),
        .s_axis_b_tready(s_axis_b_tready),

        .m_axis_result_tvalid(axis_ab_tvalid),
        .m_axis_result_tdata( axis_ab_tdata),
        .m_axis_result_tready(axis_ab_tready)
    );

    // latency 12
    floating_point_add fl_add (
        .aclk(ap_aclk),

        .s_axis_a_tvalid(prev_valid_out),
        .s_axis_a_tdata(prev_buffer_out),
        .s_axis_a_tready(s_axis_c_in_tready),

        .s_axis_b_tvalid(axis_ab_tvalid),
        .s_axis_b_tdata( axis_ab_tdata),
        .s_axis_b_tready(axis_ab_tready),

        .m_axis_result_tvalid(m_axis_c_out_tvalid),
        .m_axis_result_tdata(m_axis_c_out_tdata),
        .m_axis_result_tready(m_axis_c_out_tready)
    );

    assign m_axis_a_reg_out_tdata = a_buffer_out;
    assign m_axis_a_reg_out_tvalid = a_buffer_out;
''',
        language=dace.Language.SystemVerilog)

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
        "CONFIG.C_Rate": "1"
    })

    rtl_tasklet.add_ip_core(
    'floating_point_add', 'floating_point', 'xilinx.com', '7.1', {
        "CONFIG.Add_Sub_Value": "Add",
        "CONFIG.Axi_Optimize_Goal": "Performance",
        "CONFIG.C_Latency": "14"
    })

    # Connecting RTL Tasklet
    # In
    state.add_edge(r_A_stream, None, rtl_tasklet, "a", dace.Memlet("A_stream[0]"))
    state.add_edge(r_B_stream, None, rtl_tasklet, "b", dace.Memlet("B_stream[0]"))
    state.add_edge(r_output_stream, None, rtl_tasklet, "c_in", dace.Memlet("output_stream[0]"))

    # Out
    state.add_edge(rtl_tasklet, "a_reg_out", w_a_buff_stream, None, dace.Memlet("a_buff_stream[0]"))
    state.add_edge(rtl_tasklet, "c_out", w_c_out_stream, None, dace.Memlet("c_out_stream[0]"))

    return state


def make_sdfg(specialized):

    if specialized:
        sdfg = dace.SDFG("rtl_mm_{}x{}x{}".format(
            N.get(), K.get(), M.get()))
    else:
        sdfg = dace.SDFG("rtl_mm_NxKx{}".format(M.get()))

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

