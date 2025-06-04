module top(
    input                   clk,
                            rst_n,
                                                      
    output logic    [6:0]   Dspout, 
    output logic    [3:0]   Segout                             
);

logic s_clk;

clk_div_1hz C1(
    .clk_100mhz(clk),
    .rst_n(rst_n),
    .clk_1hz(s_clk)
);

riscv inst(
    .clk(s_clk),
    .s_clk(clk),
    .rst_n(rst_n),
    
    .Dspout(Dspout),
    .Segout(Segout)
);

endmodule

module clk_div_1hz #(
    // For a 100 MHz source, half-period = 1 s � 100 MHz = 50_000_000 cycles
    localparam integer HALF_PERIOD    = 100_000,
    localparam integer COUNTER_WIDTH  = $clog2(HALF_PERIOD)
)(
    input                           clk_100mhz, rst_n, 
    output logic                    clk_1hz      // 1 Hz global clock
);

    // Counter and raw divider signal
    reg [COUNTER_WIDTH-1:0] counter;

    // Divide logic: toggle div_clk every HALF_PERIOD cycles
    always_ff @(posedge clk_100mhz or negedge rst_n) begin
        if (!rst_n) begin
            counter <= '0;
            clk_1hz <= 1'b0;
        end else if (counter == HALF_PERIOD-1) begin
            counter <= '0;
            clk_1hz <= ~clk_1hz;
        end else begin
            counter <= counter + 1;
        end
    end

endmodule

module riscv(
    input                   clk, s_clk,
                            rst_n,   

    output logic    [6:0]   Dspout, 
    output logic    [3:0]   Segout                      
);
    logic [31:0]    addresult;
    logic [31:0]    result;
    logic [31:0]    pc_next,
                    pc_current;
    PC PC (
        .clk(clk),
        .rst_n(rst_n),
        .pc_next(pc_next),
        .pc_current(pc_current)
    );

    logic [31:0]        f_pc;
    always_ff @(posedge clk) begin 
        if (!rst_n) begin
            f_pc            <= #1 0;
        end
        else begin
            f_pc            <= #1 pc_current;    
        end      
    end

    PC_Adder PC_Adder (
        .operand1(pc_current),
        .result(result)
    );                

    logic [31:0]    instr;
    Instruction_memory Instr_Mem (
        .addr(pc_current),
        .instr(instr)
    );

    logic [31:0]    if_reg;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            if_reg          <= #1 0;
        end
        else begin
            if_reg          <= #1 instr; 
        end  
    end

    logic           branch,
                    alu_src,
                    regwrite,
                    memread,
                    memwrite,
                    mem2reg;

    logic [1:0]     alu_op;
    controller Ctrl (
        .opcode(if_reg[6:0]),

        .branch(branch),
        .alu_src(alu_src),
        .regwrite(regwrite),
        .memread(memread),
        .memwrite(memwrite),
        .memtoreg(mem2reg),

        .alu_op(alu_op)
    );
    
    logic [4:0]     waddr;
    logic           reg_wr,
                    mem_rd,
                    mem_wr,
                    mr_reg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin    
            waddr           <= #1 0;
            reg_wr          <= #1 0;
            mem_rd          <= #1 0;
            mem_wr          <= #1 0;
            mr_reg          <= #1 0;
        end
        else begin
            waddr           <= #1 if_reg[11:7];
            reg_wr          <= #1 regwrite;
            mem_rd          <= #1 memread;
            mem_wr          <= #1 memwrite;
            mr_reg          <= #1 mem2reg;
        end
    end

    logic [31:0]    wr_data, x1,x2,x3,
                    reg_data1,
                    reg_data2;
                    
    Registers inst_reg (
        .reg_addr1(if_reg[19:15]),
        .reg_addr2(if_reg[24:20]),
        .wr_addr(waddr),
        .clk(clk),
        .rst_n(rst_n),
        .wr_data(wr_data),
        .reg_data1(reg_data1),
        .reg_data2(reg_data2),
        .wr_en(reg_wr),
        
        .x1(x1),
        .x2(x2),
        .x3(x3)
    );

    logic [31:0]    imm_val;
    imm_gen immgen (
        .instr(if_reg),
        .imm_val(imm_val)
    );

    logic [3:0]     operation; 
    ALU_Ctrl ALUctrl (
        .alu_op(alu_op),
        .funct73({if_reg[30],if_reg[14:12]}),
        .operation(operation)
    );

    //logic [31:0]    fAreg, fBreg;
    logic [31:0]    imm_out;
    Mux_2x1 Imm_MUX (
        .MemtoReg(alu_src),
        .in1(reg_data2),
        .in2(imm_val),
        .out(imm_out)
    );

    logic           zero;
    logic [31:0]    ALUresult;
    ALU ALU (
        .operand1(reg_data1),
        .operand2(imm_out),
        .ALUoperation(operation),
        .ALUresult(ALUresult),
        .zero(zero)
    );
    
    Add Offset (
        .operand1(f_pc),
        .operand2(imm_val),
        .result(addresult)
    );

    logic [31:0]    alu_reg,
                    wrd_mem;               
                    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            alu_reg         <= #1 0; 
            wrd_mem         <= #1 0;
        end
        else begin
            alu_reg         <= #1 ALUresult;  
            wrd_mem         <= #1 reg_data2;  
        end
    end

    Mux_2x1 Br_MUX (
        .MemtoReg(branch & zero),
        .in1(result),
        .in2(addresult),
        .out(pc_next)
    );   

    /*
    logic           fA, fB;
    forward_unit inst_fwd_unit (
        .regwr(reg_wr),
        .rs1(if_reg[19:15]),
        .rs2(if_reg[24:20]),
        .rd(waddr),

        .fwd_A(fA),
        .fwd_B(fB)
    );

    Mux_2x1 fMux_A (
        .MemtoReg(fA),
        .in1(reg_data1),
        .in2(wr_data),
        .out(fAreg)
    );

    Mux_2x1 fMux_B (
        .MemtoReg(fB),
        .in1(reg_data2),
        .in2(wr_data),
        .out(fBreg)
    );
    */

    logic [31:0]    read, m0;
    Data_memory Data_Mem (
        .clk(clk), .wr_en(mem_wr), .rd_en(mem_rd),
        .addr(alu_reg), .wr_data(wrd_mem),
        .mem0(m0),
        .rdata(read),
        .rst_n(rst_n)
    );

    Mux_2x1 Reg_MUX (
        .MemtoReg(mr_reg),
        .in1(alu_reg),
        .in2(read),
        .out(wr_data)
    ); 

    SSD Test_Block (
        .clk_i(s_clk),

        .x1(x1[3:0]),
        .x2(x2[3:0]),
        .x3(x3[3:0]),
        .m0(m0[3:0]),

        .Dspout(Dspout),
        .Segout(Segout)  
    );

endmodule