// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Self-checking simulation testbench. Not yet executed in this environment.

`timescale 1ns/1ps
`default_nettype none

module tb_hdc_bind_xor;
    localparam int unsigned DATA_W = 256;

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

    always #5 clk = ~clk;

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
        check_xor(256'h0123456789abcdef_fedcba9876543210_0123456789abcdef_fedcba9876543210,
                  256'hffff0000ffff0000_0000ffff0000ffff_aaaa5555aaaa5555_5555aaaa5555aaaa,
                  "structured-pattern");

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
        out_ready = 1'b0;
        @(negedge clk);
        operand_a = 256'h1111111111111111_2222222222222222_3333333333333333_4444444444444444;
        operand_b = 256'h0101010101010101_0202020202020202_0303030303030303_0404040404040404;
        in_valid = 1'b1;

        @(posedge clk); // accept first word
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure-first: missing valid");
        if (result !== (operand_a ^ operand_b)) $fatal(1, "backpressure-first: wrong result");
        
        operand_a = 256'hffffffff00000000_ffffffff00000000_ffffffff00000000_ffffffff00000000;
        operand_b = 256'h00000000ffffffff_00000000ffffffff_00000000ffffffff_00000000ffffffff;
        if (in_ready !== 1'b0) $fatal(1, "backpressure: accepted while output was blocked");

        @(posedge clk); // output still blocked; second input must not replace it
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure: valid dropped while stalled");
        if (result !== 256'h1010101010101010_2020202020202020_3030303030303030_4040404040404040)
            $fatal(1, "backpressure: result changed while stalled");
        if (in_ready !== 1'b0) $fatal(1, "backpressure: ready asserted while full");

        out_ready = 1'b1; // consume first and accept second on same edge
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure-second: missing valid");
        if (result !== (operand_a ^ operand_b)) $fatal(1, "backpressure-second: wrong result");

        in_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b0) $fatal(1, "backpressure-second: output did not retire");

        $display("PASS: hdc_bind_xor smoke and backpressure tests completed");
        $finish;
    end
endmodule

`default_nettype wire
