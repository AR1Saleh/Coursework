
module ALU(
    input logic [3:0] ALUoperation,
    input logic [31:0] operand1,operand2,
    output logic [31:0] ALUresult,
    output logic zero
    );
    always_comb begin
        case (ALUoperation)
        4'b0000: ALUresult = operand1 & operand2;
        4'b0001: ALUresult = operand1 | operand2;
        4'b0010: ALUresult = operand1 + operand2;
        4'b0110: ALUresult = operand1 - operand2;
        default: ALUresult = 32'b0;
        endcase

        if (ALUoperation == 4'b0110 && ALUresult == 32'b0) begin
            zero = 1'b1;
        end 
        else begin
            zero = 1'b0;
        end
    end
endmodule

