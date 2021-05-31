# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#
#        ap_aclk               s_axis_a             s_axis_b           s_axis_c_in
#           │                     │                    │                    │
#           │                     │                    │                    │
#           │                     │32 bit              │64 bit              │64 bit
#           │               ┌─────▼──────┐             │                    │
#           │               │ Data       │             │                    │
#           │               │            │             │                    │
#           │               │ Doubler    │             │                    │
#           │               └─────┬──────┘             │                    │
#           │                     │64 bit              │                    │
#           │1x Freq              │                    │                    │
#  ┌────────┼────────────┬────────│──────────┬─────────│──────────┐         │
#  │        │            │        │          │         │          │         │
#  │ ┌──────▼─────┐      │  ┌─────▼──────┐   │   ┌─────▼──────┐   │   ┌─────▼──────┐
#  │ │Clock       │      └─►│            │   └──►│            │   └──►│            │
#  │ │            │2x Freq  │ Clock Sync │       │ Clock Sync │       │ Clock Sync │
#  │ │ Multiplier ├──────┬─►│            │   ┌──►│            │   ┌──►│            │
#  │ └────────────┘      │  └─────┬──────┘   │   └─────┬──────┘   │   └─────┬──────┘
#  │                     │        │          │         │          │         │
#  │                     ├────────│──────────┼─────────│──────────┤         │
#  │                     │        │          │         │          │         │
#  │                     │        │64 bit    │         │64 bit    │         │64 bit
#  │                     │  ┌─────▼──────┐   │   ┌─────▼──────┐   │   ┌─────▼──────┐
#  │                     │  │            │   │   │            │   │   │            │
#  │                     ├─►│ Data Issue │   └──►│ Data Issue │   └──►│ Data Issue │
#  │                     │  │            │       │            │       │            │
#  │                     │  └──────┬─────┘       └─────┬──────┘       └─────┬──────┘
#  │                     │         │32 bit             │                    │
#  │                     │         │     ┌─────────────┘                    │
#  │                     │         │     │                                  │
#  │                     │         │     │                                  │
#  │                     │     ┌───▼─────▼──┐                               │
#  │                     │     │            │                               │
#  │                     ├────►│ Multiplier │                               │
#  │                     │     │            │                               │
#  │                     │     └───────┬────┘                               │
#  │                     │             │                                    │
#  │                     │             │    ┌───────────────────────────────┘
#  │                     │             │    │
#  │                     │         ┌───▼────▼───┐
#  │                     │         │            │
#  │                     ├────────►│   Adder    │
#  │                     │         │            │
#  │                     │         └─────┬──────┘
#  │                     │               │
#  │                     │               │
#  │                     │               │32 bit
#  │                     │         ┌─────▼──────┐
#  │                     │         │ Data       │
#  │                     ├────────►│            │
#  │                     │         │ Packer     │
#  │                     │         └─────┬──────┘
#  │                     │               │
#  │                     │               │64 bit
#  │                     │         ┌─────▼──────┐
#  │                     └────────►│            │
#  │                               │ Clock Sync │
#  └──────────────────────────────►│            │
#                                  └─────┬──────┘
#                                        │
#                                        │
#                                        │
#                                        │
#                                        ▼
#                                   m_axis_c_out

import argparse
import dace
import numpy as np
import pdb
import select
import sys

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")

sdfg = dace.SDFG("rtl_mm_double_pump_p")

# vectorization
veclen = 2
sdfg.add_constant('VECLEN', veclen)

def make_copy_to_fpga_state(sdfg):

    ###########################################################################
    # Copy data to FPGA

    state = sdfg.add_state("copy_to_device", is_start_state=True)

    sdfg.add_array("A", [N, K],
                    dtype=dace.float32)
    sdfg.add_array("B", [K, M // veclen],
                    dtype=dace.vector(dace.float32, veclen))
    sdfg.add_array("C", [N, M // veclen],
                    dtype=dace.vector(dace.float32, veclen))

    A_host = state.add_read("A")
    B_host = state.add_read("B")

    sdfg.add_array("A_device", [N, K],
                   dtype=dace.float32,
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)
    sdfg.add_array("B_device", [K, M // veclen],
                   dtype=dace.vector(dace.float32, veclen),
                   transient=True,
                   storage=dace.StorageType.FPGA_Global)

    A_device = state.add_write("A_device")
    B_device = state.add_write("B_device")

    state.add_memlet_path(A_host, A_device, memlet=dace.Memlet("A_device"))
    state.add_memlet_path(B_host, B_device, memlet=dace.Memlet("B_device[0:K, 0:M//VECLEN]"))

    return state


def make_copy_to_host_state(sdfg):

    ###########################################################################
    # Copy data to host

    state = sdfg.add_state("copy_to_host")

    Output_buffer = state.add_read("output_buffer")
    C_host = state.add_write("C")

    state.add_memlet_path(Output_buffer, C_host, memlet=dace.Memlet("C[0:N, 0:M//VECLEN]"))

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

        base_clk_freq = dace.Config.get('compiler', 'xilinx', 'frequency')
        if base_clk_freq == '':
            base_clk_freq='300'
        double_clk_freq = str(2 * int(base_clk_freq))

        rtl_tasklet = state.add_tasklet(
            name="rtl_ma",
            inputs={"a", "b", "c_in"},
            outputs={"c_out"},
            code='''
        assign ap_done = 1; // free-running

        wire clk_sp;
        wire clk_dp;
        wire rstn_sp;
        wire rstn_dp;

        clk_wiz_0 clock_multiplier (
            .clk_in1(ap_aclk),
            .clk_out1(clk_sp),
            .clk_out2(clk_dp)
        );

        rst_clk_wiz rst_clk_wiz_sp (
            .slowest_sync_clk(clk_sp),
            .ext_reset_in(ap_areset),
            .peripheral_aresetn(rstn_sp),
            .aux_reset_in(0),
            .dcm_locked(1),
            .mb_debug_sys_rst(0)
        );

        rst_clk_wiz rst_clk_wiz_dp (
            .slowest_sync_clk(clk_dp),
            .ext_reset_in(ap_areset),
            .peripheral_aresetn(rstn_dp),
            .aux_reset_in(0),
            .dcm_locked(1),
            .mb_debug_sys_rst(0)
        );

        // Doubeling the a stream to issue at double freq
        wire        axis_a_tvalid;
        wire [63:0] axis_a_tdata;
        wire        axis_a_tready;

        assign axis_a_tvalid = s_axis_a_tvalid;
        assign axis_a_tdata = {s_axis_a_tdata, s_axis_a_tdata};
        assign s_axis_a_tready = axis_a_tready;


        wire        axis_a_dpclk_tvalid;
        wire [63:0] axis_a_dpclk_tdata;
        wire        axis_a_dpclk_tready;

        slow_to_fast_clk clock_sync_a (
            .s_axis_aclk(clk_sp),
            .s_axis_aresetn(rstn_sp),
            .m_axis_aclk(clk_dp),
            .m_axis_aresetn(rstn_dp),

            .s_axis_tvalid(axis_a_tvalid),
            .s_axis_tdata(axis_a_tdata),
            .s_axis_tready(axis_a_tready),

            .m_axis_tvalid(axis_a_dpclk_tvalid),
            .m_axis_tdata(axis_a_dpclk_tdata),
            .m_axis_tready(axis_a_dpclk_tready)
        );

        wire        axis_a_dp_tvalid;
        wire [31:0] axis_a_dp_tdata;
        wire        axis_a_dp_tready;

        slow_to_fast_data data_issue_a (
            .aclk(clk_dp),
            .aresetn(rstn_dp),

            .s_axis_tvalid(axis_a_dpclk_tvalid),
            .s_axis_tdata(axis_a_dpclk_tdata),
            .s_axis_tready(axis_a_dpclk_tready),

            .m_axis_tvalid(axis_a_dp_tvalid),
            .m_axis_tdata(axis_a_dp_tdata),
            .m_axis_tready(axis_a_dp_tready)
        );

        wire        axis_b_dpclk_tvalid;
        wire [63:0] axis_b_dpclk_tdata;
        wire        axis_b_dpclk_tready;

        slow_to_fast_clk clock_sync_b (
            .s_axis_aclk(clk_sp),
            .s_axis_aresetn(rstn_sp),
            .m_axis_aclk(clk_dp),
            .m_axis_aresetn(rstn_dp),

            .s_axis_tvalid(s_axis_b_tvalid),
            .s_axis_tdata(s_axis_b_tdata),
            .s_axis_tready(s_axis_b_tready),

            .m_axis_tvalid(axis_b_dpclk_tvalid),
            .m_axis_tdata(axis_b_dpclk_tdata),
            .m_axis_tready(axis_b_dpclk_tready)
        );

        wire        axis_b_dp_tvalid;
        wire [31:0] axis_b_dp_tdata;
        wire        axis_b_dp_tready;

        slow_to_fast_data data_issue_b (
            .aclk(clk_dp),
            .aresetn(rstn_dp),

            .s_axis_tvalid(axis_b_dpclk_tvalid),
            .s_axis_tdata(axis_b_dpclk_tdata),
            .s_axis_tready(axis_b_dpclk_tready),

            .m_axis_tvalid(axis_b_dp_tvalid),
            .m_axis_tdata(axis_b_dp_tdata),
            .m_axis_tready(axis_b_dp_tready)
        );

        wire        axis_ab_tvalid;
        wire [31:0] axis_ab_tdata;
        wire        axis_ab_tready;

        floating_point_mult fl_mult (
            .aclk(clk_dp),
            .aresetn(rstn_dp),

            .s_axis_a_tvalid(axis_a_dp_tvalid),
            .s_axis_a_tdata(axis_a_dp_tdata),
            .s_axis_a_tready(axis_a_dp_tready),

            .s_axis_b_tvalid(axis_b_dp_tvalid),
            .s_axis_b_tdata(axis_b_dp_tdata),
            .s_axis_b_tready(axis_b_dp_tready),

            .m_axis_result_tvalid(axis_ab_tvalid),
            .m_axis_result_tdata(axis_ab_tdata),
            .m_axis_result_tready(axis_ab_tready)
        );

        wire        axis_c_in_dpclk_tvalid;
        wire [63:0] axis_c_in_dpclk_tdata;
        wire        axis_c_in_dpclk_tready;

        slow_to_fast_clk clock_sync_c_in (
            .s_axis_aclk(clk_sp),
            .s_axis_aresetn(rstn_sp),
            .m_axis_aclk(clk_dp),
            .m_axis_aresetn(rstn_dp),

            .s_axis_tvalid(s_axis_c_in_tvalid),
            .s_axis_tdata(s_axis_c_in_tdata),
            .s_axis_tready(s_axis_c_in_tready),

            .m_axis_tvalid(axis_c_in_dpclk_tvalid),
            .m_axis_tdata(axis_c_in_dpclk_tdata),
            .m_axis_tready(axis_c_in_dpclk_tready)
        );

        wire        axis_c_in_dp_tvalid;
        wire [31:0] axis_c_in_dp_tdata;
        wire        axis_c_in_dp_tready;

        slow_to_fast_data data_issue_c (
            .aclk(clk_dp),
            .aresetn(rstn_dp),

            .s_axis_tvalid(axis_c_in_dpclk_tvalid),
            .s_axis_tdata(axis_c_in_dpclk_tdata),
            .s_axis_tready(axis_c_in_dpclk_tready),

            .m_axis_tvalid(axis_c_in_dp_tvalid),
            .m_axis_tdata(axis_c_in_dp_tdata),
            .m_axis_tready(axis_c_in_dp_tready)
        );

        wire        axis_c_out_dp_tvalid;
        wire [31:0] axis_c_out_dp_tdata;
        wire        axis_c_out_dp_tready;

        floating_point_add fl_add (
            .aclk(clk_dp),
            .aresetn(rstn_dp),

            .s_axis_a_tvalid(axis_c_in_dp_tvalid),
            .s_axis_a_tdata(axis_c_in_dp_tdata),
            .s_axis_a_tready(axis_c_in_dp_tready),

            .s_axis_b_tvalid(axis_ab_tvalid),
            .s_axis_b_tdata(axis_ab_tdata),
            .s_axis_b_tready(axis_ab_tready),

            .m_axis_result_tvalid(axis_c_out_dp_tvalid),
            .m_axis_result_tdata(axis_c_out_dp_tdata),
            .m_axis_result_tready(axis_c_out_dp_tready)
        );

        wire        axis_c_out_dpclk_tvalid;
        wire [63:0] axis_c_out_dpclk_tdata;
        wire        axis_c_out_dpclk_tready;

        fast_to_slow_data data_packer_c_out (
            .aclk(clk_dp),
            .aresetn(rstn_dp),

            .s_axis_tvalid(axis_c_out_dp_tvalid),
            .s_axis_tdata(axis_c_out_dp_tdata),
            .s_axis_tready(axis_c_out_dp_tready),

            .m_axis_tvalid(axis_c_out_dpclk_tvalid),
            .m_axis_tdata(axis_c_out_dpclk_tdata),
            .m_axis_tready(axis_c_out_dpclk_tready)
        );

        fast_to_slow_clk clock_sync_result (
            .s_axis_aclk(clk_dp),
            .s_axis_aresetn(rstn_dp),
            .m_axis_aclk(clk_sp),
            .m_axis_aresetn(rstn_sp),

            .s_axis_tvalid(axis_c_out_dpclk_tvalid),
            .s_axis_tdata(axis_c_out_dpclk_tdata),
            .s_axis_tready(axis_c_out_dpclk_tready),

            .m_axis_tvalid(m_axis_c_out_tvalid),
            .m_axis_tdata(m_axis_c_out_tdata),
            .m_axis_tready(m_axis_c_out_tready)
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

        rtl_tasklet.add_ip_core(
            "clk_wiz_0", "clk_wiz", "xilinx.com", "6.0", {
                "CONFIG.PRIMITIVE": "Auto",
                "CONFIG.PRIM_IN_FREQ": base_clk_freq,
                "CONFIG.CLKOUT2_USED": "true",
                "CONFIG.CLKOUT1_REQUESTED_OUT_FREQ": base_clk_freq,
                "CONFIG.CLKOUT2_REQUESTED_OUT_FREQ": double_clk_freq,
                "CONFIG.CLKIN1_JITTER_PS": "33.330000000000005",
                "CONFIG.CLKOUT1_DRIVES": "Buffer",
                "CONFIG.CLKOUT2_DRIVES": "Buffer",
                "CONFIG.CLKOUT3_DRIVES": "Buffer",
                "CONFIG.CLKOUT4_DRIVES": "Buffer",
                "CONFIG.CLKOUT5_DRIVES": "Buffer",
                "CONFIG.CLKOUT6_DRIVES": "Buffer",
                "CONFIG.CLKOUT7_DRIVES": "Buffer",
                "CONFIG.FEEDBACK_SOURCE": "FDBK_AUTO",
                "CONFIG.USE_LOCKED": "false",
                "CONFIG.USE_RESET": "false",
                "CONFIG.MMCM_DIVCLK_DIVIDE": "1",
                "CONFIG.MMCM_BANDWIDTH": "OPTIMIZED",
                "CONFIG.MMCM_CLKFBOUT_MULT_F": "4",
                "CONFIG.MMCM_CLKIN1_PERIOD": "3.333",
                "CONFIG.MMCM_CLKIN2_PERIOD": "10.0",
                "CONFIG.MMCM_COMPENSATION": "AUTO",
                "CONFIG.MMCM_CLKOUT0_DIVIDE_F": "4",
                "CONFIG.MMCM_CLKOUT1_DIVIDE": "2",
                "CONFIG.NUM_OUT_CLKS": "2",
                "CONFIG.CLKOUT1_JITTER": "81.814",
                "CONFIG.CLKOUT1_PHASE_ERROR": "77.836",
                "CONFIG.CLKOUT2_JITTER": "71.438",
                "CONFIG.CLKOUT2_PHASE_ERROR": "77.836",
                "CONFIG.AUTO_PRIMITIVE": "PLL"
            })
        
        rtl_tasklet.add_ip_core('rst_clk_wiz', 'proc_sys_reset', 'xilinx.com', '5.0',
                                {})

        rtl_tasklet.add_ip_core('slow_to_fast_clk', 'axis_clock_converter',
                                'xilinx.com', '1.1', {
                                    "CONFIG.TDATA_NUM_BYTES": "8",
                                    "CONFIG.SYNCHRONIZATION_STAGES": "8"
                                })

        rtl_tasklet.add_ip_core('fast_to_slow_clk', 'axis_clock_converter',
                                'xilinx.com', '1.1', {
                                    "CONFIG.TDATA_NUM_BYTES": "8",
                                    "CONFIG.SYNCHRONIZATION_STAGES": "8"
                                })

        rtl_tasklet.add_ip_core('slow_to_fast_data', 'axis_dwidth_converter',
                                'xilinx.com', '1.1', {
                                    "CONFIG.S_TDATA_NUM_BYTES": "8",
                                    "CONFIG.M_TDATA_NUM_BYTES": "4"
                                })

        rtl_tasklet.add_ip_core('fast_to_slow_data', 'axis_dwidth_converter',
                                'xilinx.com', '1.1', {
                                    "CONFIG.S_TDATA_NUM_BYTES": "4",
                                    "CONFIG.M_TDATA_NUM_BYTES": "8"
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

    pre_state = make_copy_to_fpga_state(sdfg)
    compute_state = make_fpga_state(sdfg, p)
    post_state = make_copy_to_host_state(sdfg)

    sdfg.add_edge(pre_state, compute_state, dace.InterstateEdge())
    sdfg.add_edge(compute_state, post_state, dace.InterstateEdge())

    return sdfg

def probe_max_freq():
    from dace.codegen import exceptions as cgx
    
    min = 100
    max = 200

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

    #sdfg = make_sdfg()
    #testing = 1000 #(min+max)//2
    #dace.Config.set('compiler', 'xilinx', 'frequency', value=str(testing))
    #print (f"~~~~ Test Frequency: {testing} ~~~")
    #print(dace.Config.get('compiler', 'xilinx', 'frequency'))
    #sdfg.compile()
    #print(dace.Config.get('compiler', 'xilinx', 'frequency'))

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
#        M.get(), N.get(), K.get()))

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

#    #print("--- A ---")
#    #print(A)
#    #print("*** B ***")
#    #print(B)
#    #print("=== C ===")
#    #print(C)
#    #print("=========")

#    diff = np.linalg.norm((A @ B) - C) / float(M.get() * K.get())
#    if diff > 1e-6:
#        raise ValueError(f"Verification failed, difference: {diff}")
#    else:
#        print("Results successfully verified.")