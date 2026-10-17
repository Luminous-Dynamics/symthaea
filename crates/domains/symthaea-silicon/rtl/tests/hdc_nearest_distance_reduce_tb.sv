`timescale 1ns/1ps

module hdc_nearest_distance_reduce_tb;
    localparam int DIST_WIDTH = 15;
    localparam int ITEM_ID_WIDTH = 32;

    logic clk = 1'b0;
    logic rst_n = 1'b0;

    logic s_valid = 1'b0;
    logic s_ready;
    logic [DIST_WIDTH-1:0] s_distance = '0;
    logic [ITEM_ID_WIDTH-1:0] s_item_id = '0;
    logic s_last = 1'b0;

    logic m_valid;
    logic m_ready = 1'b0;
    logic [DIST_WIDTH-1:0] m_distance;
    logic [ITEM_ID_WIDTH-1:0] m_item_id;

    always #5 clk = ~clk;

    hdc_nearest_distance_reduce #(
        .DIST_WIDTH(DIST_WIDTH),
        .ITEM_ID_WIDTH(ITEM_ID_WIDTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .s_valid_i(s_valid),
        .s_ready_o(s_ready),
        .s_distance_i(s_distance),
        .s_item_id_i(s_item_id),
        .s_last_i(s_last),
        .m_valid_o(m_valid),
        .m_ready_i(m_ready),
        .m_distance_o(m_distance),
        .m_item_id_o(m_item_id)
    );

    task automatic send_candidate(
        input [DIST_WIDTH-1:0] distance_value,
        input [ITEM_ID_WIDTH-1:0] item_id_value,
        input last_value
    );
        begin
            @(negedge clk);
            while (!s_ready) @(negedge clk);
            s_valid = 1'b1;
            s_distance = distance_value;
            s_item_id = item_id_value;
            s_last = last_value;
            @(posedge clk); // s_ready is high for this accepted beat.
            @(negedge clk);
            s_valid = 1'b0;
            s_last = 1'b0;
        end
    endtask

    task automatic check_and_consume(
        input [DIST_WIDTH-1:0] expected_distance,
        input [ITEM_ID_WIDTH-1:0] expected_item_id
    );
        integer stall_cycle;
        begin
            wait (m_valid === 1'b1);
            if (m_distance !== expected_distance || m_item_id !== expected_item_id)
                $fatal(1, "wrong result: distance=%0d id=%0d expected_distance=%0d expected_id=%0d",
                       m_distance, m_item_id, expected_distance, expected_item_id);

            // Hold downstream not-ready and prove both payload and valid remain stable.
            for (stall_cycle = 0; stall_cycle < 3; stall_cycle = stall_cycle + 1) begin
                @(posedge clk);
                #1;
                if (m_valid !== 1'b1 ||
                    m_distance !== expected_distance ||
                    m_item_id !== expected_item_id)
                    $fatal(1, "output changed during backpressure at stall cycle %0d",
                           stall_cycle);
                if (s_ready !== 1'b0)
                    $fatal(1, "input was not backpressured while result stalled");
            end

            @(negedge clk);
            m_ready = 1'b1;
            @(posedge clk);
            #1;
            if (m_valid !== 1'b0)
                $fatal(1, "result did not clear after handshake");
            @(negedge clk);
            m_ready = 1'b0;
        end
    endtask

    initial begin
        #12;
        rst_n = 1'b1;

        // Repeated minimum: smallest stable ID must win independently of stream order.
        send_candidate(15'd19, 32'd5, 1'b0);
        send_candidate(15'd7,  32'd9, 1'b0);
        send_candidate(15'd7,  32'd2, 1'b0);
        send_candidate(15'd11, 32'd1, 1'b1);
        check_and_consume(15'd7, 32'd2);

        // A one-candidate query at the maximum legal 16,384-bit Hamming distance.
        send_candidate(15'd16384, 32'd11, 1'b1);
        check_and_consume(15'd16384, 32'd11);

        // Zero is a valid distance, and a tie still uses the stable-ID policy.
        send_candidate(15'd0, 32'd8, 1'b0);
        send_candidate(15'd0, 32'd4, 1'b1);
        check_and_consume(15'd0, 32'd4);

        $display("PASS: hdc_nearest_distance_reduce directed testbench");
        $finish;
    end

    // A broken handshake must not turn the test into an infinite wait.
    initial begin
        #100000;
        $fatal(1, "testbench timeout");
    end
endmodule
