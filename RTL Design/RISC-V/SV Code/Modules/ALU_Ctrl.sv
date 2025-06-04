module ALU_Ctrl (
    input  logic [1:0]             alu_op,
    input  logic [3:0]             funct73,         
    output logic [3:0]             operation      
);

always_comb begin : Ctrl
    case(alu_op)
        2'b00: operation = 4'b0010;          // ADD
        2'b01: operation = 4'b0110;          // SUB
        
        2'b10: begin
            case(funct73)
                4'b0000: operation = 4'b0010; // ADD
                4'b1000: operation = 4'b0110; // SUB
                4'b0111: operation = 4'b0000; // AND
                4'b0110: operation = 4'b0001; // OR
                default: operation = 4'bx;    // Undefined funct73
            endcase
        end
        default: operation = 4'bx;           // Undefined alu_op
    endcase
end

endmodule