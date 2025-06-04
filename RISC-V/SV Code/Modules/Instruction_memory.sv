module Instruction_memory(
    input  logic [31:0] addr,
    output logic [31:0] instr
);
    logic [31:0] mem [63:0]; 
    // Right‑shift by 2 to index words in a byte‑addressed scheme
    assign instr = mem[addr >> 2];

endmodule
