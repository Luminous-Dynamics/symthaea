// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Self-checking Hamming-count testbench. Not yet executed in this environment.

`timescale 1ns/1ps
`default_nettype none

module tb_hdc_hamming_count #(
    parameter int unsigned DATA_W = 256,
    parameter int unsigned COUNT_W = $clog2(DATA_W + 1),
    parameter bit BALANCED_TREE = 1'b0
);

    logic clk = 1'b0;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [DATA_W-1:0] operand_a;
    logic [DATA_W-1:0] operand_b;
    logic out_valid;
    logic out_ready;
    logic [COUNT_W-1:0] distance;
    logic [31:0] rng_a;
    logic [31:0] rng_b;
    logic [DATA_W-1:0] random_a;
    logic [DATA_W-1:0] random_b;
    logic [DATA_W-1:0] pattern_a;
    logic [DATA_W-1:0] pattern_b;
    logic [COUNT_W-1:0] held_distance;
    localparam int unsigned COUNT_HEX_BITS = ((COUNT_W + 3) / 4) * 4;
    localparam int unsigned ORACLE_ROW_W = (2 * DATA_W) + COUNT_HEX_BITS;
    localparam int unsigned MAX_ORACLE_WORDS = 4096;
    logic [ORACLE_ROW_W-1:0] oracle_words [0:MAX_ORACLE_WORDS-1];
    logic [ORACLE_ROW_W-1:0] oracle_row;
    integer oracle_word_count;
    string oracle_path;

    hdc_hamming_count #(.DATA_W(DATA_W), .COUNT_W(COUNT_W), .BALANCED_TREE(BALANCED_TREE)) dut (
        .clk,
        .rst_n,
        .in_valid,
        .in_ready,
        .operand_a,
        .operand_b,
        .out_valid,
        .out_ready,
        .distance
    );

    initial begin
        forever #5 clk = ~clk;
    end

    task automatic check_count_expected(
        input logic [DATA_W-1:0] a,
        input logic [DATA_W-1:0] b,
        input logic [COUNT_W-1:0] expected_oracle,
        input string label
    );
        logic [COUNT_W-1:0] expected_local;
        expected_local = COUNT_W'($countones(a ^ b));
        if (expected_oracle !== expected_local)
            $fatal(1, "%s: software oracle disagrees with $countones", label);

        @(negedge clk);
        if (in_ready !== 1'b1) $fatal(1, "%s: expected in_ready before request", label);
        operand_a = a;
        operand_b = b;
        in_valid = 1'b1;

        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "%s: missing out_valid", label);
        if (distance !== expected_oracle)
            $fatal(1, "%s: popcount mismatch got=%0d expected=%0d", label, distance, expected_oracle);

        in_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b0) $fatal(1, "%s: output did not retire", label);
    endtask

    task automatic check_count(
        input logic [DATA_W-1:0] a,
        input logic [DATA_W-1:0] b,
        input string label
    );
        check_count_expected(a, b, COUNT_W'($countones(a ^ b)), label);
    endtask

    initial begin
        rst_n = 1'b0;
        in_valid = 1'b0;
        operand_a = '0;
        operand_b = '0;
        out_ready = 1'b1;
        rng_a = 32'h13579bdf;
        rng_b = 32'h2468ace1;

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        check_count('0, '0, "distance-zero");
        check_count({DATA_W{1'b1}}, '0, "distance-full");
        check_count({DATA_W{1'b1}}, {DATA_W{1'b1}}, "ones-ones-zero");

        for (int bit_index = 0; bit_index < DATA_W; bit_index++) begin
            pattern_a[bit_index] = ((bit_index % 7) < 3);
            pattern_b[bit_index] = ((bit_index % 11) < 5);
        end
        check_count(pattern_a, pattern_b, "structured-pattern");

        for (int trial = 0; trial < 32; trial++) begin
            for (int lane = 0; lane < DATA_W; lane += 32) begin
                rng_a = rng_a * 32'd1664525 + 32'd1013904223;
                rng_b = rng_b * 32'd22695477 + 32'd1;
                random_a[lane +: 32] = rng_a;
                random_b[lane +: 32] = rng_b;
            end
            check_count(random_a, random_b, "deterministic-lcg");
        end

        // Optional canonical-BinaryHV software-oracle words. The .mem row is
        // {operand_a, operand_b, count_hex_padded}; the expected count uses only
        // the low COUNT_W bits. The software generator checks per-word counts
        // sum to BinaryHV::hamming_distance for each complete vector pair.
        if ($value$plusargs("ORACLE_HAMMING=%s", oracle_path)) begin
            if (!$value$plusargs("ORACLE_WORDS=%d", oracle_word_count))
                $fatal(1, "ORACLE_HAMMING requires +ORACLE_WORDS=<count>");
            if (oracle_word_count < 1 || oracle_word_count > MAX_ORACLE_WORDS)
                $fatal(1, "ORACLE_WORDS out of range: %0d", oracle_word_count);
            $readmemh(oracle_path, oracle_words, 0, oracle_word_count - 1);
            for (int oracle_index = 0; oracle_index < oracle_word_count; oracle_index++) begin
                oracle_row = oracle_words[oracle_index];
                check_count_expected(
                    oracle_row[ORACLE_ROW_W-1 -: DATA_W],
                    oracle_row[ORACLE_ROW_W-DATA_W-1 -: DATA_W],
                    oracle_row[COUNT_W-1:0],
                    "canonical-BinaryHV-oracle"
                );
            end
            $display("PASS: hdc_hamming_count software oracle words=%0d width=%0d balanced_tree=%0d", oracle_word_count, DATA_W, BALANCED_TREE);
        end

        // Stalled output must hold its accepted result, even if input pins move.
        pattern_a = '0;
        pattern_b = '0;
        pattern_a[0] = 1'b1;
        pattern_b[DATA_W-1] = 1'b1;
        held_distance = COUNT_W'($countones(pattern_a ^ pattern_b));

        out_ready = 1'b0;
        @(negedge clk);
        operand_a = pattern_a;
        operand_b = pattern_b;
        in_valid = 1'b1;

        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure: missing out_valid");
        if (distance !== held_distance) $fatal(1, "backpressure: initial distance mismatch");

        operand_a = '0;
        operand_b = '0;
        if (in_ready !== 1'b0) $fatal(1, "backpressure: ready asserted while full");

        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure: valid dropped while stalled");
        if (distance !== held_distance) $fatal(1, "backpressure: output changed while stalled");

        out_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b1) $fatal(1, "backpressure: second output missing");
        if (distance !== '0) $fatal(1, "backpressure: second output should be zero");

        in_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (out_valid !== 1'b0) $fatal(1, "backpressure: output did not retire");

        $display("PASS: hdc_hamming_count smoke and backpressure tests completed width=%0d balanced_tree=%0d", DATA_W, BALANCED_TREE);
        $finish;
    end
endmodule

`default_nettype wire
