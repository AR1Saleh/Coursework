module PC(
        input logic clk,rst_n,
        input logic [31:0]  pc_next,
        output logic [31:0] pc_current
    );
    always @(posedge clk,negedge rst_n) begin
        if (~rst_n) begin
            pc_current <= 32'b0;
        end
        else begin
            pc_current <= pc_next;
        end
    end
endmodule
