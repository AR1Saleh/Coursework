
module imm_gen(
    input [31:0] instr,
    output logic [31:0] imm_val
    );
    logic [6:0] op_code;
    assign op_code = instr[6:0];
    
    always_comb begin
    
        case (op_code) 
            7'b0000011,//I-type load
            7'b0010011: imm_val = { {20{ instr[31] }} , instr[31:20] }; //I-type
            7'b0100011: imm_val = { {20{ instr[31] }},instr[31:25],instr[11:7] };//S-type
            7'b1100011: imm_val = {{20{instr[31]}},instr[7],instr[30:25],instr[11:8],1'b0};//B-type
            7'b0110111,//U-type LUI
            7'b0010111: imm_val = {instr[31:12],12'b0};//U-type AUIPC
            7'b1101111: imm_val = {{11{ instr[31] }},instr[31],instr[19:12],instr[20],instr[30:21],1'b0};//J-type JAL
            7'b1100111: imm_val = {{20{instr[31]}}, instr[31:20]};//JALR
            default: imm_val = 32'b0;
        endcase
          
    end
endmodule

