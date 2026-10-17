// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Protocol/equivalence bench for the parameterized ready/valid XOR stream.
// The top sweeps lane widths; reported cycles are simulation protocol counts,
// not achieved hardware frequency, energy, resource, or speedup measurements.
`timescale 1ns/1ps
`default_nettype none

module hdc_bind_xor_stream_case #(
    parameter integer LANE_WIDTH = 256
) (
    output reg done
);
    localparam integer VECTOR_BITS = 16384;
    localparam integer BEATS = VECTOR_BITS / LANE_WIDTH;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg s_valid = 1'b0;
    wire s_ready;
    reg [LANE_WIDTH-1:0] s_a = {LANE_WIDTH{1'b0}};
    reg [LANE_WIDTH-1:0] s_b = {LANE_WIDTH{1'b0}};
    reg s_last = 1'b0;

    wire m_valid;
    reg m_ready = 1'b0;
    wire [LANE_WIDTH-1:0] m_data;
    wire m_last;

    reg stalled_q = 1'b0;
    reg [LANE_WIDTH-1:0] held_data_q;
    reg held_last_q;

    reg [VECTOR_BITS-1:0] vector_a;
    reg [VECTOR_BITS-1:0] vector_b;
    integer bit_index;

    always #5 clk = ~clk;

    hdc_bind_xor_stream #(.DATA_WIDTH(LANE_WIDTH)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_valid(s_valid),
        .s_ready(s_ready),
        .s_a(s_a),
        .s_b(s_b),
        .s_last(s_last),
        .m_valid(m_valid),
        .m_ready(m_ready),
        .m_data(m_data),
        .m_last(m_last)
    );

    // An output beat held under backpressure must remain valid and bit-stable.
    always @(posedge clk) begin
        if (!rst_n) begin
            stalled_q <= 1'b0;
        end else begin
            if (stalled_q) begin
                if (m_valid !== 1'b1 || m_data !== held_data_q || m_last !== held_last_q)
                    $fatal(1, "lane=%0d: output changed while stalled", LANE_WIDTH);
            end
            if (m_valid && !m_ready) begin
                stalled_q <= 1'b1;
                held_data_q <= m_data;
                held_last_q <= m_last;
            end else begin
                stalled_q <= 1'b0;
            end
        end
    end

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

    task automatic collect_vector(
        output reg [VECTOR_BITS-1:0] observed,
        output integer cycles
    );
        integer beat;
        begin
            observed = {VECTOR_BITS{1'b0}};
            beat = 0;
            cycles = 0;
            while (beat < BEATS) begin
                @(negedge clk);
                // Deterministic downstream backpressure: both short and
                // longer pauses, repeated identically for each lane profile.
                m_ready = ((cycles % 5) != 2) && ((cycles % 7) != 4);
                @(posedge clk);
                cycles = cycles + 1;
                if (m_valid && m_ready) begin
                    observed[beat*LANE_WIDTH +: LANE_WIDTH] = m_data;
                    if (m_last !== (beat == BEATS - 1))
                        $fatal(1, "lane=%0d beat=%0d: last marker mismatch", LANE_WIDTH, beat);
                    beat = beat + 1;
                end
                if (cycles > (BEATS * 10 + 100))
                    $fatal(1, "lane=%0d: timeout waiting for output", LANE_WIDTH);
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
        reg [VECTOR_BITS-1:0] expected;
        reg [VECTOR_BITS-1:0] observed;
        integer vector_cycles;
        begin
            expected = in_a ^ in_b;
            observed = {VECTOR_BITS{1'b0}};
            vector_cycles = 0;
            fork
                send_vector(in_a, in_b);
                collect_vector(observed, vector_cycles);
            join
            if (observed !== expected)
                $fatal(1, "lane=%0d vector=%0d: reconstructed 16384-bit output mismatch", LANE_WIDTH, vector_id);
            $display(
                "PROFILE lane_bits=%0d beats=%0d vector_id=%0d simulated_cycles=%0d deterministic_stalls=enabled exact_output=PASS",
                LANE_WIDTH, BEATS, vector_id, vector_cycles
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

        // Edge case: all-zero XOR all-one.
        vector_a = {VECTOR_BITS{1'b0}};
        vector_b = {VECTOR_BITS{1'b1}};
        run_vector(vector_a, vector_b, 0);

        // Deterministic irregular data pattern, fixed across all lane widths.
        for (bit_index = 0; bit_index < VECTOR_BITS; bit_index = bit_index + 1) begin
            vector_a[bit_index] = ((bit_index * 13 + 7) % 11) < 5;
            vector_b[bit_index] = ((bit_index * 17 + 3) % 13) < 7;
        end
        run_vector(vector_a, vector_b, 1);

        // Self-bind cancellation must reconstruct the all-zero hypervector.
        run_vector(vector_a, vector_a, 2);

        done = 1'b1;
    end
endmodule

module hdc_bind_xor_stream_tb;
    wire done64;
    wire done128;
    wire done256;
    wire done512;
    wire done16384;

    hdc_bind_xor_stream_case #(.LANE_WIDTH(64)) c64 (.done(done64));
    hdc_bind_xor_stream_case #(.LANE_WIDTH(128)) c128 (.done(done128));
    hdc_bind_xor_stream_case #(.LANE_WIDTH(256)) c256 (.done(done256));
    hdc_bind_xor_stream_case #(.LANE_WIDTH(512)) c512 (.done(done512));
    hdc_bind_xor_stream_case #(.LANE_WIDTH(16384)) c16384 (.done(done16384));

    initial begin
        wait (done64 && done128 && done256 && done512 && done16384);
        $display("PASS: hdc_bind_xor_stream_tb; lanes=64,128,256,512,16384; exact reconstructed outputs and deterministic backpressure checks completed");
        $finish;
    end
endmodule

`default_nettype wire
