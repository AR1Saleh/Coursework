module Registers (
  input                 clk, rst_n, wr_en,

  input         [4:0]   reg_addr1,
                        reg_addr2,
                        wr_addr,

  input         [31:0]  wr_data,

  output logic  [31:0]  reg_data1,
                        reg_data2

);
  logic [31:0] regfile [31:0];
  integer i;

  // synchronous reset, synchronous write
  always @(negedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < 32; i = i + 1)
        regfile[i] <= 32'b0;
    end else if (wr_en && (wr_addr != 0)) begin
      // typical RISC‑V style: x0 stays zero, all other regs are writable
      regfile[wr_addr] <= wr_data;
    end
  end

  // combinational read ports, and hard‑wire read of x0 = 0
  assign reg_data1 = (reg_addr1 == 0) ? 32'b0 : regfile[reg_addr1];
  assign reg_data2 = (reg_addr2 == 0) ? 32'b0 : regfile[reg_addr2];

endmodule
