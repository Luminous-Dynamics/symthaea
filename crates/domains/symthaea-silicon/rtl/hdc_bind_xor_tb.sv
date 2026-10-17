// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Directed simulation bench for the exact XOR-bind RTL primitive.
// The per-bit walk covers every input position on each port.
`timescale 1ns/1ps

module hdc_bind_xor_tb;
    localparam integer WIDTH = 16384;

    reg  [WIDTH-1:0] a;
    reg  [WIDTH-1:0] b;
    wire [WIDTH-1:0] y;
    reg  [WIDTH-1:0] expected;
    integer i;

    hdc_bind_xor #(.WIDTH(WIDTH)) dut (
        .a(a),
        .b(b),
        .y(y)
    );

    task automatic check_case(input string case_name);
        begin
            #1;
            if (y !== expected) begin
                $display("FAIL: %s", case_name);
                $fatal(1);
            end
        end
    endtask

    initial begin
        // Boundary patterns.
        a = '0;
        b = '0;
        expected = '0;
        check_case("zero XOR zero");

        a = '1;
        b = '0;
        expected = '1;
        check_case("ones XOR zero");

        a = '1;
        b = '1;
        expected = '0;
        check_case("ones XOR ones");

        // Complementary structured pattern must produce all ones.
        for (i = 0; i < WIDTH; i = i + 1) begin
            a[i] = ((i % 7) < 3);
            b[i] = ~a[i];
        end
        expected = '1;
        check_case("complementary structured patterns");

        // Full-width deterministic irregular pattern. Expected values are
        // independently populated bit by bit from the driven operands.
        for (i = 0; i < WIDTH; i = i + 1) begin
            a[i] = ((i * 13 + 7) % 11) < 5;
            b[i] = ((i * 17 + 3) % 13) < 7;
            expected[i] = a[i] ^ b[i];
        end
        check_case("deterministic full-width pattern");

        // Exercise every position on input A, input B, and self-bind
        // cancellation. This catches bit slicing/indexing mistakes across
        // the full 16,384-bit width, not just the least-significant lanes.
        for (i = 0; i < WIDTH; i = i + 1) begin
            a = '0;
            b = '0;
            expected = '0;
            a[i] = 1'b1;
            expected[i] = 1'b1;
            check_case("walk-one on input A");

            a = '0;
            b = '0;
            expected = '0;
            b[i] = 1'b1;
            expected[i] = 1'b1;
            check_case("walk-one on input B");

            a = '0;
            b = '0;
            a[i] = 1'b1;
            b[i] = 1'b1;
            expected = '0;
            check_case("same-bit self-bind cancellation");
        end

        $display("PASS: hdc_bind_xor_tb; WIDTH=%0d; all directed and per-bit checks completed", WIDTH);
        $finish;
    end
endmodule
