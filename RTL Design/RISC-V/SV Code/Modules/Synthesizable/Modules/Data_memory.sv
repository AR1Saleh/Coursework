
module Data_memory(
    input clk,wr_en,rd_en,rst_n,
    input [31:0] addr,[31:0] wr_data,    
    output logic [31:0] rdata, 
    output logic [31:0] mem0
    );

int i;
logic [31:0] data_mem [1:0];
assign rdata = (rd_en) ? data_mem[addr]: 32'b0;
assign mem0 = data_mem[0];

always @(negedge clk) begin
    if (!rst_n) begin
      for (i=0;i<2;i++) begin
        data_mem[i] <= 32'b0;
      end  
    end
    else if (wr_en) begin
      data_mem[addr] <= wr_data;
    end
end
endmodule

