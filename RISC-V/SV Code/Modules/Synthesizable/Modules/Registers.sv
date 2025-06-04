module Registers(
  input                 clk, rst_n, wr_en,

  input         [4:0]   reg_addr1,
                        reg_addr2,
                        wr_addr,

  input         [31:0]  wr_data,

  output logic  [31:0]  reg_data1,
                        reg_data2,
                        x1, x2, x3
);
  logic [31:0] regfile [31:0];
  integer      i;

  // single clocked block: async reset, synchronous write
  always @(negedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // clear all regs including regfile[0]
      for (i = 0; i < 32; i = i + 1)
        regfile[i] <= 32'b0;
    end else begin
      // only write on wr_en and never write address 0
      if (wr_en && (wr_addr != 0))
        regfile[wr_addr] <= wr_data;
    end
  end

  // read ports clamp address 0 to zero
  assign reg_data1 = (reg_addr1 == 0) ? 32'b0 : regfile[reg_addr1];
  assign reg_data2 = (reg_addr2 == 0) ? 32'b0 : regfile[reg_addr2];

  // expose x1, x2, x3 if you want
  assign x1 = regfile[1];
  assign x2 = regfile[2];
  assign x3 = regfile[3];

endmodule
