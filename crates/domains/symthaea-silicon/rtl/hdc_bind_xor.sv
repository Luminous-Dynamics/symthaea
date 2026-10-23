// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One-word exact HDC bind/unbind primitive.
// This is a streaming datapath slice, not a complete 16,384-bit vector controller,
// SoC, CPU, GPU, or qualified accelerator.

`timescale 1ns/1ps
`default_nettype none

module hdc_bind_xor #(
    parameter int unsigned DATA_W = 256
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [DATA_W-1:0]     operand_a,
    input  logic [DATA_W-1:0]     operand_b,

    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [DATA_W-1:0]     result
);

    // Single-entry elastic output register. When a consumer applies
    // backpressure, both valid and result remain stable until accepted.
    assign in_ready = (~out_valid) | out_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            result    <= '0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) begin
                result <= operand_a ^ operand_b;
            end
        end
    end

endmodule

`default_nettype wire
