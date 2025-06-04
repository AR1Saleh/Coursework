module controller(
    input           [6:0]   opcode,
    output logic            branch, alu_src, regwrite, memread, memwrite, memtoreg,
    output logic    [1:0]   alu_op
);
    always_comb begin : ctrl
        case (opcode)
            3: begin        // Loads
                alu_src     = 'b1;
                memtoreg    = 'b1;
                regwrite    = 'b1;
                memread     = 'b1;
                memwrite    = 'b0;
                branch      = 'b0;
                alu_op      = 'b00;
            end
            19: begin       // I-type
                alu_src     = 'b1;
                memtoreg    = 'b0;
                regwrite    = 'b1;
                memread     = 'b0;
                memwrite    = 'b0;
                branch      = 'b0;
                alu_op      = 'b00;
            end
            35: begin       // Stores
                alu_src     = 'b1;
                //memtoreg    = 'bx;
                regwrite    = 'b0;
                memread     = 'b0;
                memwrite    = 'b1;
                branch      = 'b0;
                alu_op      = 'b00;
            end
            51: begin       // R-type
                alu_src     = 'b0;
                memtoreg    = 'b0;
                regwrite    = 'b1;
                memread     = 'b0;
                memwrite    = 'b0;
                branch      = 'b0;
                alu_op      = 'b10;
            end
            99: begin       // Branches
                alu_src     = 'b0;
                //memtoreg    = 'bx;
                regwrite    = 'b0;
                memread     = 'b0;
                memwrite    = 'b0;
                branch      = 'b1;
                alu_op      = 'b01;
            end
            default: begin
                alu_src     = 'bx;
                memtoreg    = 'bx;
                regwrite    = 'bx;
                memread     = 'bx;
                memwrite    = 'bx;
                branch      = 'bx;
                alu_op      = 'bxx;
            end
        endcase
    end
endmodule