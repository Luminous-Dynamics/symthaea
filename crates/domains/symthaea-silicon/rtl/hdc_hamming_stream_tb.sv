// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Exactness/protocol bench for streaming Hamming distance.
// Reference count is independently accumulated bit-by-bit in the testbench;
// canonical Rust oracle differential remains a separate required gate.
\`timescale 1ns/1ps
\`default_nettype none

module hdc_hamming_stream_case #(
    parameter integer LANE_WIDTH = 256
) (
    output reg done
);
    localparam integer VECTOR_BITS = 16384;
    localparam integer BEATS = VECTOR_BITS / LANE_WIDTH;
    localparam integer COUNT_WIDTH = $clog2(VECTOR_BITS + 1);

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg s_valid = 1'b0;
    wire s_ready;
    reg [LANE_WIDTH-1:0] s_a = {LANE_WIDTH{1'b0}};
    reg [LANE_WIDTH-1:0] s_b = {LANE_WIDTH{1'b0}};
    reg s_last = 1'b0;

    wire m_valid;
    reg m_ready = 1'b0;
    wire [COUNT_WIDTH-1:0] m_distance;
    wire protocol_error;

    reg stalled_q = 1'b0;
    reg [COUNT_WIDTH-1:0] held_distance_q;
    reg [VECTOR_BITS-1:0] vector_a;
    reg [VECTOR_BITS-1:0] vector_b;
    integer bit_index;

    always #5 clk = ~clk;

    hdc_hamming_stream #(
        .DATA_WIDTH(LANE_WIDTH),
        .VECTOR_BITS(VECTOR_BITS),
        .COUNT_WIDTH(COUNT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_valid(s_valid),
        .s_ready(s_ready),
        .s_a(s_a),
        .s_b(s_b),
        .s_last(s_last),
        .m_valid(m_valid),
        .m_ready(m_ready),
        .m_distance(m_distance),
        .protocol_error(protocol_error)
    );

    // A completed distance held under backpressure must remain stable.
    always @(posedge clk) begin
        if (!rst_n) begin
            stalled_q <= 1'b0;
        end else begin
            if (stalled_q) begin
                if (m_valid !== 1'b1 || m_distance !== held_distance_q)
                    $fatal(1, "lane=%0d: distance result changed while stalled", LANE_WIDTH);
            end
            if (m_valid && !m_ready) begin
                stalled_q <= 1'b1;
                held_distance_q <= m_distance;
            end else begin
                stalled_q <= 1'b0;
            end
            if (protocol_error)
                $fatal(1, "lane=%0d: valid-frame test unexpectedly raised protocol_error", LANE_WIDTH);
        end
    end

    function [COUNT_WIDTH-1:0] reference_distance;
        input [VECTOR_BITS-1:0] a;
        input [VECTOR_BITS-1:0] b;
        integer k;
        begin
            reference_distance = {COUNT_WIDTH{1'b0}};
            for (k = 0; k < VECTOR_BITS; k = k + 1)
                reference_distance = reference_distance + (a[k] ^ b[k]);
        end
    endfunction

    task automatic send_vector(
        input reg [VECTOR_BITS-1:0] in_a,
        input reg [VECTOR_BITS-1:0] in_b
    );
        integer beat;
        begin
            for (beat = 0; beat < BEATS; beat = beat + 1) begin
                @(negedge clk);
                s_valid = 1'b1;
                s_a = in_a[beat*LANE_WIDTH +: LANE_WIDTH];
                s_b = in_b[beat*LANE_WIDTH +: LANE_WIDTH];
                s_last = (beat == BEATS - 1);
                do begin
                    @(posedge clk);
                end while (!s_ready);
            end
            @(negedge clk);
            s_valid = 1'b0;
            s_last = 1'b0;
            s_a = {LANE_WIDTH{1'b0}};
            s_b = {LANE_WIDTH{1'b0}};
        end
    endtask

    task automatic collect_result(
        output reg [COUNT_WIDTH-1:0] observed,
        output integer cycles
    );
        integer waits;
        begin
            observed = {COUNT_WIDTH{1'b0}};
            cycles = 0;
            waits = 0;
            while (1) begin
                @(negedge clk);
                m_ready = ((waits % 5) != 2) && ((waits % 7) != 4);
                @(posedge clk);
                waits = waits + 1;
                cycles = cycles + 1;
                if (m_valid && m_ready) begin
                    observed = m_distance;
                    // The stream produces exactly one result for this vector.
                    break;
                end
                if (waits > (BEATS * 10 + 100))
                    $fatal(1, "lane=%0d: timeout waiting for distance result", LANE_WIDTH);
            end
            @(negedge clk);
            m_ready = 1'b0;
        end
    endtask

    task automatic run_vector(
        input reg [VECTOR_BITS-1:0] in_a,
        input reg [VECTOR_BITS-1:0] in_b,
        input integer vector_id
    );
        reg [COUNT_WIDTH-1:0] expected;
        reg [COUNT_WIDTH-1:0] observed;
        integer vector_cycles;
        begin
            expected = reference_distance(in_a, in_b);
            observed = {COUNT_WIDTH{1'b0}};
            vector_cycles = 0;
            fork
                send_vector(in_a, in_b);
                collect_result(observed, vector_cycles);
            join
            if (observed !== expected)
                $fatal(1, "lane=%0d vector=%0d: distance mismatch expected=%0d got=%0d", LANE_WIDTH, vector_id, expected, observed);
            $display(
                "HAMMING_PROFILE lane_bits=%0d beats=%0d vector_id=%0d expected_distance=%0d simulated_cycles=%0d deterministic_stalls=enabled exact_result=PASS",
                LANE_WIDTH, BEATS, vector_id, observed, vector_cycles
            );
        end
    endtask

    initial begin
        done = 1'b0;
        vector_a = {VECTOR_BITS{1'b0}};
        vector_b = {VECTOR_BITS{1'b0}};
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Identical vectors: distance zero.
        vector_a = {VECTOR_BITS{1'b0}};
        vector_b = {VECTOR_BITS{1'b0}};
        run_vector(vector_a, vector_b, 0);

        // Fully opposite bit patterns: maximum representable distance.
        vector_a = {VECTOR_BITS{1'b0}};
        vector_b = {VECTOR_BITS{1'b1}};
        run_vector(vector_a, vector_b, 1);

        // Fixed deterministic pattern, shared across the lane-width sweep.
        for (bit_index = 0; bit_index < VECTOR_BITS; bit_index = bit_index + 1) begin
            vector_a[bit_index] = ((bit_index * 13 + 7) % 11) < 5;
            vector_b[bit_index] = ((bit_index * 17 + 3) % 13) < 7;
        end
        run_vector(vector_a, vector_b, 2);
        run_vector(vector_a, vector_a, 3);

        done = 1'b1;
    end
endmodule

module hdc_hamming_stream_invalid_frame_tb (
    output reg done
);
    localparam integer VECTOR_BITS = 16384;
    localparam integer DATA_WIDTH = 256;
    localparam integer BEATS = VECTOR_BITS / DATA_WIDTH;
    localparam integer COUNT_WIDTH = $clog2(VECTOR_BITS + 1);

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg s_valid = 1'b0;
    wire s_ready;
    reg [DATA_WIDTH-1:0] s_a = {DATA_WIDTH{1'b0}};
    reg [DATA_WIDTH-1:0] s_b = {DATA_WIDTH{1'b1}};
    reg s_last = 1'b0;
    wire m_valid;
    reg m_ready = 1'b1;
    wire [COUNT_WIDTH-1:0] m_distance;
    wire protocol_error;
    integer beat;

    always #5 clk = ~clk;

    hdc_hamming_stream #(
        .DATA_WIDTH(DATA_WIDTH),
        .VECTOR_BITS(VECTOR_BITS),
        .COUNT_WIDTH(COUNT_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_ready(s_ready),
        .s_a(s_a), .s_b(s_b), .s_last(s_last),
        .m_valid(m_valid), .m_ready(m_ready),
        .m_distance(m_distance), .protocol_error(protocol_error)
    );

    task automatic reset_dut;
        begin
            @(negedge clk);
            rst_n = 1'b0;
            s_valid = 1'b0;
            s_last = 1'b0;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
        end
    endtask

    task automatic expect_protocol_error(input reg [8*24-1:0] label_text);
        begin
            repeat (2) @(posedge clk);
            if (protocol_error !== 1'b1 || m_valid !== 1'b0 || s_ready !== 1'b0)
                $fatal(1, "malformed frame was not rejected: %0s", label_text);
            $display("NEGATIVE_CONTROL rejected=%0s protocol_error=PASS no_result=PASS", label_text);
        end
    endtask

    initial begin
        done = 1'b0;
        repeat (3) @(posedge clk);
        reset_dut();

        // Negative control 1: early last on the first of 64 beats.
        @(negedge clk);
        s_valid = 1'b1;
        s_last = 1'b1;
        do begin
            @(posedge clk);
        end while (!s_ready);
        @(negedge clk);
        s_valid = 1'b0;
        s_last = 1'b0;
        expect_protocol_error("early_last");

        // Clear sticky framing fault with reset before the second negative.
        reset_dut();

        // Negative control 2: omit last on the expected final beat.
        for (beat = 0; beat < BEATS; beat = beat + 1) begin
            @(negedge clk);
            s_valid = 1'b1;
            s_last = 1'b0;
            s_a = {DATA_WIDTH{1'b0}};
            s_b = {DATA_WIDTH{1'b1}};
            do begin
                @(posedge clk);
            end while (!s_ready);
        end
        @(negedge clk);
        s_valid = 1'b0;
        s_last = 1'b0;
        expect_protocol_error("missing_last");

        done = 1'b1;
    end
endmodule

module hdc_hamming_stream_tb;
    wire done64;
    wire done128;
    wire done256;
    wire done512;
    wire done16384;
    wire done_invalid;

    hdc_hamming_stream_case #(.LANE_WIDTH(64)) c64 (.done(done64));
    hdc_hamming_stream_case #(.LANE_WIDTH(128)) c128 (.done(done128));
    hdc_hamming_stream_case #(.LANE_WIDTH(256)) c256 (.done(done256));
    hdc_hamming_stream_case #(.LANE_WIDTH(512)) c512 (.done(done512));
    hdc_hamming_stream_case #(.LANE_WIDTH(16384)) c16384 (.done(done16384));
    hdc_hamming_stream_invalid_frame_tb invalid_frames (.done(done_invalid));

    initial begin
        wait (done64 && done128 && done256 && done512 && done16384 && done_invalid);
        $display("PASS: hdc_hamming_stream_tb; lanes=64,128,256,512,16384; exact distances, backpressure and malformed-frame rejection completed");
        $finish;
    end
endmodule

\`default_nettype wire
