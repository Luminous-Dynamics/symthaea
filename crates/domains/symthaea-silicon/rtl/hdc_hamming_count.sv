// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Exact word-level Hamming count primitive for BinaryHV similarity.
// This is one slice of a future 16,384-bit controller/accumulator, not a full
// similarity engine or qualified accelerator.

`timescale 1ns/1ps
`default_nettype none

module hdc_hamming_count #(
    parameter int unsigned DATA_W = 256,
    parameter int unsigned COUNT_W = $clog2(DATA_W + 1),
    parameter bit BALANCED_TREE = 1'b0
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [DATA_W-1:0]       operand_a,
    input  logic [DATA_W-1:0]       operand_b,

    output logic                    out_valid,
    input  logic                    out_ready,
    output logic [COUNT_W-1:0]      distance
);

    logic [COUNT_W-1:0] popcount_comb;
    localparam int unsigned LEVELS = (DATA_W > 1) ? $clog2(DATA_W) : 0;

    // Two equivalent architectural baselines: a linear combinational
    // accumulation chain, or a balanced adder tree. Selection is a parameter
    // so synthesis comparisons bind the same exact operation and test vectors.
    generate
        if (BALANCED_TREE) begin : g_balanced_tree
            logic [COUNT_W-1:0] tree [0:LEVELS][0:DATA_W-1];

            for (genvar bit_index = 0; bit_index < DATA_W; bit_index++) begin : g_leaf
                assign tree[0][bit_index] =
                    {{(COUNT_W-1){1'b0}}, (operand_a[bit_index] ^ operand_b[bit_index])};
            end

            for (genvar layer = 0; layer < LEVELS; layer++) begin : g_layer
                localparam int unsigned WIDTH_AT_LEVEL =
                    (DATA_W + (1 << layer) - 1) >> layer;
                localparam int unsigned NEXT_WIDTH = (WIDTH_AT_LEVEL + 1) >> 1;

                for (genvar node = 0; node < NEXT_WIDTH; node++) begin : g_node
                    if ((2 * node + 1) < WIDTH_AT_LEVEL) begin : g_pair
                        assign tree[layer+1][node] =
                            tree[layer][2*node] + tree[layer][2*node+1];
                    end else begin : g_odd_carry
                        assign tree[layer+1][node] = tree[layer][2*node];
                    end
                end
            end

            assign popcount_comb = tree[LEVELS][0];
        end else begin : g_linear
            always_comb begin
                popcount_comb = '0;
                for (int unsigned bit_index = 0; bit_index < DATA_W; bit_index++) begin
                    popcount_comb = popcount_comb
                        + {{(COUNT_W-1){1'b0}}, (operand_a[bit_index] ^ operand_b[bit_index])};
                end
            end
        end
    endgenerate

    // Single-entry elastic output register; valid/data hold under backpressure.
    assign in_ready = (~out_valid) | out_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            distance  <= '0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) begin
                distance <= popcount_comb;
            end
        end
    end

endmodule

`default_nettype wire
