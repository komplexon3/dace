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

    Output_buffer = state.add_read("output_buffer")
    C_host = state.add_write("C")

    state.add_edge(Output_buffer, None, C_host, None, dace.Memlet("C"))

    return state

def make_fpga_state(sdfg):

    state = sdfg.add_state("rtl_mm")

    A = state.add_read("A_device")
    B = state.add_read("B_device")

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

    sdfg.add_array("output_buffer", [N,M],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    r_output_buffer = state.add_read("output_buffer")
    w_output_buffer = state.add_write("output_buffer")

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

    sdfg.add_scalar("a_reg",
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Local)
    r_a_reg = state.add_read("a_reg")
    w_a_reg = state.add_write("a_reg")


    ###########################################################################
    ### Process input 
    # In a map
    in_entry, in_exit = state.add_map(
        "in_map",
        dict(n="0:N", k="0:K", m="0:M"),
        schedule=dace.ScheduleType.FPGA_Device)

    # Input a processing tasklet
    read_in = state.add_tasklet("read_in",
                                {"a_in", "a_reg_in", "b_in", "out_in"},
                                {"a_out", "a_reg_out", "b_out", "out_out"},
                                """
a = a_in if m == 0 else a_reg_in # there is still always a read from the array in the generated code so this ternary statement can probably be omitted
a_out = a
a_reg_out = a
b_out = b_in
out_out = 0 if k == 0 else out_in
                            """)

    state.add_memlet_path(A,
                          in_entry,
                          read_in,
                          dst_conn="a_in",
                          memlet=dace.Memlet("A_device[n, k]"))
    state.add_memlet_path(r_a_reg,
                          in_entry,
                          read_in,
                          dst_conn="a_reg_in",
                          memlet=dace.Memlet("a_reg[0]"))
    state.add_memlet_path(read_in,
                          in_exit,
                          w_a_reg,
                          src_conn="a_reg_out",
                          memlet=dace.Memlet("a_reg[0]"))
    state.add_memlet_path(read_in,
                          in_exit,
                          w_A_stream,
                          src_conn="a_out",
                          memlet=dace.Memlet("A_stream[0]"))

    state.add_memlet_path(B,
                          in_entry,
                          read_in,
                          dst_conn="b_in",
                          memlet=dace.Memlet("B_device[k, m]"))
    state.add_memlet_path(read_in,
                          in_exit,
                          w_B_stream,
                          src_conn="b_out",
                          memlet=dace.Memlet("B_stream[0]"))

    state.add_memlet_path(r_output_buffer,
                          in_entry,
                          read_in,
                          dst_conn="out_in",
                          memlet=dace.Memlet("output_buffer[n,m]"))
    state.add_memlet_path(read_in,
                          in_exit,
                          w_output_stream,
                          src_conn="out_out",
                          memlet=dace.Memlet("output_stream[0]"))


#    # In b map
#    in_b_entry, in_b_exit = state.add_map(
#        "in_b_map",
#        dict(n="0:N", k="0:K", m="0:M"),
#        schedule=dace.ScheduleType.FPGA_Device)

#    # Input processing tasklet
#    read_b_in = state.add_tasklet("read_b_in",
#                                {"b_in"},
#                                {"b_out"},
#                                """
#b_out = b_in
#                                """)

#    state.add_memlet_path(B,
#                          in_b_entry,
#                          read_b_in,
#                          dst_conn="b_in",
#                          memlet=dace.Memlet("B_device[k, m]"))
#    state.add_memlet_path(read_b_in,
#                          in_b_exit,
#                          w_B_stream,
#                          src_conn="b_out",
#                          memlet=dace.Memlet("B_stream[0]"))

#    # In output buffer map
#    in_buff_entry, in_buff_exit = state.add_map(
#        "in_buff_map",
#        dict(n="0:N", k="0:K", m="0:M"),
#        schedule=dace.ScheduleType.FPGA_Device)

#    # Input processing tasklet
#    read_buff_in = state.add_tasklet("read_buff_in",
#                                {"out_in"},
#                                {"out_out"},
#                                """
#out_out = 0 if k == 0 else out_in
#                                """)

#    state.add_memlet_path(r_output_buffer,
#                          in_buff_entry,
#                          read_buff_in,
#                          dst_conn="out_in",
#                          memlet=dace.Memlet("output_buffer[n,m]"))
#    state.add_memlet_path(read_buff_in,
#                          in_buff_exit,
#                          w_output_stream,
#                          src_conn="out_out",
#                          memlet=dace.Memlet("output_stream[0]"))

    ###########################################################################
    ### Process c_out -> output buffer 
    # Map
    out_entry, out_exit = state.add_map(
        "out_map",
        dict(n="0:N", k="0:K", m="0:M"),
        schedule=dace.ScheduleType.FPGA_Device)

    proc_out = state.add_tasklet("proc_out", {"c_out"}, {"out_buf"}, "out_buf = c_out")

    state.add_memlet_path(r_c_out_stream,
                          out_entry,
                          proc_out,
                          dst_conn="c_out",
                          memlet=dace.Memlet("c_out_stream[0]"))
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
        name="rtl_ma",
        inputs={"a", "b", "c_in"},
        outputs={"c_out"},
        code='''
    assign ap_done = 1; // free-running
    wire ap_aresetn = ~ap_areset;

    wire        axis_ab_tvalid;
    wire [31:0] axis_ab_tdata;
    wire        axis_ab_tready;

    /* FIFO Queue for c_in https://github.com/alexforencich/verilog-axis/blob/master/rtl/axis_pipeline_register.v */
    localparam LENGTH = 9;
    /*
    * AXI input
    */
    wire [31:0]  s_axis_buff_tdata;
    wire         s_axis_buff_tvalid;
    wire         s_axis_buff_tready;

    /*
    * AXI output
    */
    wire [31:0]  m_axis_buff_tdata;
    wire         m_axis_buff_tvalid;
    wire         m_axis_buff_tready;

    wire [31:0]  axis_tdata[0:LENGTH];
    wire         axis_tvalid[0:LENGTH];
    wire         axis_tready[0:LENGTH];

    assign axis_tdata[0] = s_axis_buff_tdata;
    assign axis_tvalid[0] = s_axis_buff_tvalid;
    assign s_axis_buff_tready = axis_tready[0];

    assign m_axis_buff_tdata = axis_tdata[LENGTH];
    assign m_axis_buff_tvalid = axis_tvalid[LENGTH];
    assign axis_tready[LENGTH] = m_axis_buff_tready;

    generate
        genvar i;
        for (i = 0; i < LENGTH; i = i + 1) begin : pipe_reg
            wire [31:0]          s_axis_tdata;
            wire                 s_axis_tvalid;
            wire                 s_axis_tready;
            wire [31:0]          m_axis_tdata;
            wire                 m_axis_tvalid;
            wire                 m_axis_tready;

            assign s_axis_tdata  = axis_tdata[i];
            assign s_axis_tvalid = axis_tvalid[i];
            assign s_axis_tready = axis_tready[i];

            assign m_axis_tdata  = axis_tdata[i+1];
            assign m_axis_tvalid = axis_tvalid[i+1];
            assign m_axis_tready = axis_tready[i+1];

            reg                  s_axis_tready_reg = 1'b0;

            reg [31:0]           m_axis_tdata_reg  = {31{1'b0}};
            reg                  m_axis_tvalid_reg = 1'b0, m_axis_tvalid_next;
            reg                  m_axis_tlast_reg  = 1'b0;

            reg [31:0]           temp_m_axis_tdata_reg  = {31{1'b0}};
            reg                  temp_m_axis_tvalid_reg = 1'b0, temp_m_axis_tvalid_next;
            reg                  temp_m_axis_tlast_reg  = 1'b0;

            // datapath control
            reg store_axis_input_to_output;
            reg store_axis_input_to_temp;
            reg store_axis_temp_to_output;

            assign s_axis_tready = s_axis_tready_reg;

            assign m_axis_tdata  = m_axis_tdata_reg;
            assign m_axis_tvalid = m_axis_tvalid_reg;

            // enable ready input next cycle if output is ready or the temp reg will not be filled on the next cycle (output reg empty or no input)
            wire s_axis_tready_early = m_axis_tready || (!temp_m_axis_tvalid_reg && (!m_axis_tvalid_reg || !s_axis_tvalid));

            always @* begin
                // transfer sink ready state to source
                m_axis_tvalid_next = m_axis_tvalid_reg;
                temp_m_axis_tvalid_next = temp_m_axis_tvalid_reg;

                store_axis_input_to_output = 1'b0;
                store_axis_input_to_temp = 1'b0;
                store_axis_temp_to_output = 1'b0;

                if (s_axis_tready_reg) begin
                    // input is ready
                    if (m_axis_tready || !m_axis_tvalid_reg) begin
                        // output is ready or currently not valid, transfer data to output
                        m_axis_tvalid_next = s_axis_tvalid;
                        store_axis_input_to_output = 1'b1;
                    end else begin
                        // output is not ready, store input in temp
                        temp_m_axis_tvalid_next = s_axis_tvalid;
                        store_axis_input_to_temp = 1'b1;
                    end
                end else if (m_axis_tready) begin
                    // input is not ready, but output is ready
                    m_axis_tvalid_next = temp_m_axis_tvalid_reg;
                    temp_m_axis_tvalid_next = 1'b0;
                    store_axis_temp_to_output = 1'b1;
                end
            end

            always @(posedge ap_aclk) begin
                if (ap_areset) begin
                    s_axis_tready_reg <= 1'b0;
                    m_axis_tvalid_reg <= 1'b0;
                    temp_m_axis_tvalid_reg <= 1'b0;
                end else begin
                    s_axis_tready_reg <= s_axis_tready_early;
                    m_axis_tvalid_reg <= m_axis_tvalid_next;
                    temp_m_axis_tvalid_reg <= temp_m_axis_tvalid_next;
                end

                // datapath
                if (store_axis_input_to_output) begin
                    m_axis_tdata_reg <= s_axis_tdata;
                end else if (store_axis_temp_to_output) begin
                    m_axis_tdata_reg <= temp_m_axis_tdata_reg;
                end

                if (store_axis_input_to_temp) begin
                    temp_m_axis_tdata_reg <= s_axis_tdata;
                end
            end

        end
    endgenerate
    /* End FIFO Queue */

    /* Connect FIFO In*/
    assign s_axis_buff_tdata  = s_axis_c_in_tdata;
    assign s_axis_buff_tvalid = s_axis_c_in_tvalid;
    assign s_axis_buff_tready = s_axis_c_in_tready;

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

        .s_axis_a_tvalid(m_axis_buff_tdata),
        .s_axis_a_tdata(m_axis_buff_tvalid),
        .s_axis_a_tready(m_axis_buff_tready),

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
    state.add_edge(r_A_stream, None, rtl_tasklet, "a", dace.Memlet("A_stream[0]"))
    state.add_edge(r_B_stream, None, rtl_tasklet, "b", dace.Memlet("B_stream[0]"))
    state.add_edge(r_output_stream, None, rtl_tasklet, "c_in", dace.Memlet("output_stream[0]"))

    # Out
    state.add_edge(rtl_tasklet, "c_out", w_c_out_stream, None, dace.Memlet("c_out_stream[0]"))

    return state


def make_sdfg():

    sdfg = dace.SDFG("rtl_matrix_multiplication_buff")

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
    args = vars(parser.parse_args())

    M.set(args["M"])
    N.set(args["N"])
    K.set(args["K"])
    sdfg = make_sdfg()

    print("Matrix multiplication {}x{}x{}".format(
        M.get(), N.get(), K.get()))

    # Initialize arrays: Randomize A and B, zero C
    A = np.ndarray([N.get(), K.get()], dtype=dace.float32.type)
    B = np.ndarray([K.get(), M.get()], dtype=dace.float32.type)
    C = np.ndarray([N.get(), M.get()], dtype=dace.float32.type)
    A[:] = np.random.rand(N.get(), K.get()).astype(dace.float32.type)
    B[:] = np.random.rand(K.get(), M.get()).astype(dace.float32.type)
    C[:] = dace.float32(0)

    A_regression = np.ndarray([N.get(), K.get()], dtype=np.float32)
    B_regression = np.ndarray([K.get(), M.get()], dtype=np.float32)
    C_regression = np.ndarray([N.get(), M.get()], dtype=np.float32)
    A_regression[:] = A[:]
    B_regression[:] = B[:]
    C_regression[:] = C[:]

    sdfg(A=A, B=B, C=C, M=M, N=N, K=K)

    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
    if diff > 1e-6:
        raise ValueError(f"Verification failed, difference: {diff}")
    else:
        print("Results successfully verified.")
