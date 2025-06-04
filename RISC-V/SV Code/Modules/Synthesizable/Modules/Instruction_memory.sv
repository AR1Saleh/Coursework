module Instruction_memory(
    input           [31:0] addr,
    output logic    [31:0] instr
);

    // 16-word instruction ROM 
    logic [31:0] mem [0:15] = '{
            // 0
            32'h00a00093,  // addi x1, x0, 10
            // 1
            32'h00102023,  // sw   x1,   0(x0)
            // 2
            32'h00002103,  // lw   x2,   0(x0)
            // 3
            32'h00100093,  // addi x1,   x0, 1
            // 4
            32'h00500113,  // addi x2,   x0, 5
            // 5
            32'h002081b3,  // add  x3,   x1, x2
            // 6  
            32'h00100093,  // addi x1,   x0, 1
            // 7  
            32'h00100113,  // addi x2,   x0, 1
            // 8  
            32'h00208463,  // beq  x1,   x2, +8
            // 9 
            32'h00108093,  // addi x1,   x1, 1
            // 10 
            32'h00108093,  // addi x1,   x1, 1
            // 11 
            32'h00100093,  // addi x1,   x0, 1
            // 12 
            32'h00102023,  // sw   x1,   0(x0)
            // 13 
            32'h00002103,  // lw   x2,   0(x0)
            // 14 
            32'h00110193,   // addi x3,   x2, 1
            // 15
            32'h00000013  // addi x0,   x0, 0  (NOP)
        };
    // byte-address ? word index
    logic [3:0] addr_reg = addr >> 2;
    assign instr = mem[addr_reg];

endmodule