// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Ready/valid streaming XOR primitive for exact BinaryHV chunks.
// This is a parameterized architectural profile, not a performance claim.
`timescale 1ns/1ps
`default_nettype none

module hdc_bind_xor_stream #(
    parameter integer DATA_WIDTH = 256
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  s_valid,
    output wire                  s_ready,
    input  wire [DATA_WIDTH-1:0] s_a,
    input  wire [DATA_WIDTH-1:0] s_b,
    input  wire                  s_last,

    output wire                  m_valid,
    input  wire                  m_ready,
    output wire [DATA_WIDTH-1:0] m_data,
    output wire                  m_last
);
    reg [DATA_WIDTH-1:0] data_q;
    reg                  valid_q;
    reg                  last_q;

    // One-entry elastic output stage. Under backpressure, s_ready is low and
    // all output registers hold stable. A consumed beat may be replaced on
    // the same edge, sustaining one accepted/emitted beat per cycle.
    assign s_ready = (~valid_q) | m_ready;
    assign m_valid = valid_q;
    assign m_data  = data_q;
    assign m_last  = last_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            data_q  <= {DATA_WIDTH{1'b0}};
            valid_q <= 1'b0;
            last_q  <= 1'b0;
        end else if (s_ready) begin
            valid_q <= s_valid;
            if (s_valid) begin
                data_q <= s_a ^ s_b;
                last_q <= s_last;
            end
        end
    end
endmodule

`default_nettype wire
