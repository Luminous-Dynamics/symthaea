// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Exact streaming Hamming distance for packed BinaryHV vectors.
// Count accumulation is local; one distance result is emitted per valid frame.
// This is a synthesizable candidate, not a performance claim.
`timescale 1ns/1ps
`default_nettype none

module hdc_hamming_stream #(
    parameter integer DATA_WIDTH = 256,
    parameter integer VECTOR_BITS = 16384,
    parameter integer COUNT_WIDTH = (VECTOR_BITS <= 1) ? 1 : $clog2(VECTOR_BITS + 1)
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
    output wire [COUNT_WIDTH-1:0] m_distance,
    output wire                  protocol_error
);
    localparam integer BEATS_PER_VECTOR = VECTOR_BITS / DATA_WIDTH;
    localparam integer BEAT_INDEX_WIDTH =
        (BEATS_PER_VECTOR <= 1) ? 1 : $clog2(BEATS_PER_VECTOR);

    reg [COUNT_WIDTH-1:0] accumulated_q;
    reg [COUNT_WIDTH-1:0] distance_q;
    reg [BEAT_INDEX_WIDTH-1:0] beat_index_q;
    reg valid_q;
    reg error_q;

    function [COUNT_WIDTH-1:0] count_ones;
        input [DATA_WIDTH-1:0] value;
        integer bit_index;
        begin
            count_ones = {COUNT_WIDTH{1'b0}};
            for (bit_index = 0; bit_index < DATA_WIDTH; bit_index = bit_index + 1)
                count_ones = count_ones + value[bit_index];
        end
    endfunction

    wire [COUNT_WIDTH-1:0] beat_distance = count_ones(s_a ^ s_b);
    wire expected_last = (beat_index_q == BEATS_PER_VECTOR - 1);

    // A malformed frame raises a sticky protocol error, suppresses its result,
    // and blocks further input until reset; partial distances never qualify.
    assign s_ready = (~error_q) & ((~valid_q) | m_ready);
    assign m_valid = valid_q;
    assign m_distance = distance_q;
    assign protocol_error = error_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            accumulated_q <= {COUNT_WIDTH{1'b0}};
            distance_q <= {COUNT_WIDTH{1'b0}};
            beat_index_q <= {BEAT_INDEX_WIDTH{1'b0}};
            valid_q <= 1'b0;
            error_q <= 1'b0;
        end else if (s_ready) begin
            if (valid_q && m_ready)
                valid_q <= 1'b0;

            if (s_valid) begin
                if (s_last != expected_last) begin
                    error_q <= 1'b1;
                    accumulated_q <= {COUNT_WIDTH{1'b0}};
                    beat_index_q <= {BEAT_INDEX_WIDTH{1'b0}};
                    valid_q <= 1'b0;
                end else if (s_last) begin
                    distance_q <= accumulated_q + beat_distance;
                    accumulated_q <= {COUNT_WIDTH{1'b0}};
                    beat_index_q <= {BEAT_INDEX_WIDTH{1'b0}};
                    valid_q <= 1'b1;
                end else begin
                    accumulated_q <= accumulated_q + beat_distance;
                    beat_index_q <= beat_index_q + 1'b1;
                    valid_q <= 1'b0;
                end
            end
        end
    end
endmodule

`default_nettype wire
