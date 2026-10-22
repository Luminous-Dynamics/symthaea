// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Self-checking simulation testbench. Not yet executed in this environment.

`timescale 1ns/1ps
`default_nettype none

module tb_hdc_bind_xor #(
    parameter int unsigned DATA_W = 256
);

    logic clk = 1'b0;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [DATA_W-1:0] operand_a;
    logic [DATA_W-1:0] operand_b;
    logic out_valid;
    logic out_ready;
    logic [DATA_W-1:0] result;
    logic [31:0] rng_a;
    logic [31:0] rng_b;
    logic [DATA_W-1:0] random_a;
    logic [DATA_W-1:0] random_b;
    logic [DATA_W-1:0] pattern_a;
    logic [DATA_W-1:0] pattern_b;
    logic [DATA_W-1:0] second_a;
    logic [DATA_W-1:0] second_b;
    logic [DATA_W-1:0] held_result;

    hdc_bind_xor #(.DATA_W(DATA_W)) dut (
        .clk,
        .rst_n,
        .in_valid,
        .in_ready,
        .operand_a,
        .operand_b,
        .out_valid,
        .out_ready,
        .result
    );

    initial begin
        forever #5 clk = ~clk;
    end

    task automatic check_xor(
        input logic [DATA_W-1:0] a,
        input logic [DATA_W-1:0] b,
        input string label
    );
        logic [DATA_W-1:0] expected;
        expected = a ^ b;

        @(negedge clk);
        if (in_ready !== 1'b1) $fatal(1, "%s: expected in_ready before request", label);
        operand_a = a;
        operand_b = b;
        in_valid = 1'b1;

        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "%s: missing out_valid", label);
        if (result !== expected) $fatal(1, "%s: XOR mismatch", label);

        in_valid = 1'b0;
        @(posedge clk);  // out_ready=1: consume the result
        @(negedge clk);
        if (out_valid !== 1'b0) $fatal(1, "%s: output did not retire", label);
    endtask

    initial begin
        rst_n     = 1'b0;
        in_valid  = 1'b0;
        operand_a = '0;
        operand_b = '0;
        out_ready = 1'b1;

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        check_xor('0, '0, "zero-zero");
        check_xor({DATA_W{1'b1}}, '0, "ones-zero");
        check_xor({DATA_W{1'b1}}, {DATA_W{1'b1}}, "ones-ones");
        if ((DATA_W % 32) != 0) $fatal(1, "DATA_W must be a multiple of 32");
        for (int bit_index = 0; bit_index < DATA_W; bit_index++) begin
            pattern_a[bit_index] = ((bit_index % 7) < 3);
            pattern_b[bit_index] = ((bit_index % 11) < 5);
        end
        check_xor(pattern_a, pattern_b, "structured-pattern");

        // Deterministic LCG-generated word pairs. This exercises many bit
        // combinations without relying on simulator-specific $urandom state.
        rng_a = 32'h13579bdf;
        rng_b = 32'h2468ace1;
        for (int trial = 0; trial < 32; trial++) begin
            for (int lane = 0; lane < DATA_W; lane += 32) begin
                rng_a = rng_a * 32'd1664525 + 32'd1013904223;
                rng_b = rng_b * 32'd22695477 + 32'd1;
                random_a[lane +: 32] = rng_a;
                random_b[lane +: 32] = rng_b;
            end
            check_xor(random_a, random_b, "deterministic-lcg");
        end

        // Backpressure must hold the first result and refuse a second input.
        for (int bit_index = 0; bit_index < DATA_W; bit_index++) begin
            pattern_a[bit_index] = ((bit_index % 3) == 0);
            pattern_b[bit_index] = ((bit_index % 5) == 1);
        end
        held_result = pattern_a ^ pattern_b;
        second_a = ~pattern_a;
        second_b = ~pattern_b;

        out_ready = 1'b0;
        @(negedge clk);
        operand_a = pattern_a;
        operand_b = pattern_b;
        in_valid = 1'b1;

        @(posedge clk); // accept first word
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure-first: missing valid");
        if (result !== held_result) $fatal(1, "backpressure-first: wrong result");

        // Replace the input pins while stalled. A compliant elastic stage
        // must hold the already accepted output word unchanged.
        operand_a = second_a;
        operand_b = second_b;
        if (in_ready !== 1'b0) $fatal(1, "backpressure: accepted while output was blocked");

        @(posedge clk); // output still blocked; second input must not replace it
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure: valid dropped while stalled");
        if (result !== held_result)
            $fatal(1, "backpressure: result changed while stalled");
        if (in_ready !== 1'b0) $fatal(1, "backpressure: ready asserted while full");

        out_ready = 1'b1; // consume first and accept second on same edge
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure-second: missing valid");
        if (result !== (second_a ^ second_b)) $fatal(1, "backpressure-second: wrong result");

        in_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b0) $fatal(1, "backpressure-second: output did not retire");

        $display("PASS: hdc_bind_xor smoke and backpressure tests completed");
        $finish;
    end
endmodule

`default_nettype wire
