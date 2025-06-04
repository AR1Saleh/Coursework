module controller(
    input      [6:0] opcode,
    output logic     branch, alu_src, regwrite, memread, memwrite, memtoreg,
    output logic [1:0] alu_op
);

always_comb begin : ctrl
    // Default assignments (avoids latches)
    alu_src   = 'b0;
    memtoreg  = 'b0;
    regwrite  = 'b0;
    memread   = 'b0;
    memwrite  = 'b0;
    branch    = 'b0;
    alu_op    = 'b00;

    case (opcode)
        7'b0000011: begin // Loads
            alu_src   = 1'b1;
            memtoreg  = 1'b1;
            regwrite  = 1'b1;
            memread   = 1'b1;
        end
        7'b0010011: begin // I-type
            alu_src   = 1'b1;
            regwrite  = 1'b1;
        end
        7'b0100011: begin // Stores
            alu_src   = 1'b1;
            memwrite  = 1'b1;
        end
        7'b0110011: begin // R-type
            regwrite  = 1'b1;
            alu_op    = 2'b10;
        end
        7'b1100011: begin // Branches
            branch    = 1'b1;
            alu_op    = 2'b01;
        end
        default: ; // Explicit default (no action needed)
    endcase
end
endmodule