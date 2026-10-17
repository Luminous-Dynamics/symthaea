// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Exact digital BinaryHV bind primitive.
// This is a combinational semantic reference, not a performance claim.
`timescale 1ns/1ps
`default_nettype none

module hdc_bind_xor #(
    parameter integer WIDTH = 16384
) (
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    output wire [WIDTH-1:0] y
);
    assign y = a ^ b;
endmodule

`default_nettype wire
