FIRRTL version 1.2.0
circuit RiscV :
  module InstructionMemory :
    input clock : Clock
    input reset : UInt<1>
    input io_addr : UInt<32> @[src/main/scala/riscvproc.scala 22:20]
    output io_instr : UInt<32> @[src/main/scala/riscvproc.scala 22:20]

    mem mem : @[src/main/scala/riscvproc.scala 26:22]
      data-type => UInt<32>
      depth => 16
      read-latency => 0
      write-latency => 1
      reader => io_instr_MPORT
      writer => MPORT
      writer => MPORT_1
      writer => MPORT_2
      writer => MPORT_3
      writer => MPORT_4
      writer => MPORT_5
      writer => MPORT_6
      writer => MPORT_7
      writer => MPORT_8
      writer => MPORT_9
      writer => MPORT_10
      writer => MPORT_11
      writer => MPORT_12
      writer => MPORT_13
      writer => MPORT_14
      writer => MPORT_15
      read-under-write => undefined
    node _io_instr_T = shr(io_addr, 2) @[src/main/scala/riscvproc.scala 36:34]
    node _io_instr_T_1 = bits(_io_instr_T, 3, 0) @[src/main/scala/riscvproc.scala 36:39]
    node defaultProgram_0 = UInt<24>("ha00093") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_1 = UInt<24>("h102023") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_2 = UInt<24>("h2103") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_3 = UInt<24>("h100093") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_4 = UInt<24>("h500113") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_5 = UInt<24>("h2081b3") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_6 = UInt<24>("h100093") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_7 = UInt<24>("h100113") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_8 = UInt<24>("h208663") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_9 = UInt<24>("h13") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_10 = UInt<24>("h13") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_11 = UInt<24>("h108093") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_12 = UInt<24>("h100093") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_13 = UInt<24>("h102023") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_14 = UInt<24>("h2103") @[src/main/scala/riscvproc.scala 27:{37,37}]
    node defaultProgram_15 = UInt<24>("h110193") @[src/main/scala/riscvproc.scala 27:{37,37}]
    io_instr <= mem.io_instr_MPORT.data @[src/main/scala/riscvproc.scala 36:18]
    mem.io_instr_MPORT.addr <= _io_instr_T_1 @[src/main/scala/riscvproc.scala 36:24]
    mem.io_instr_MPORT.en <= UInt<1>("h1") @[src/main/scala/riscvproc.scala 36:24]
    mem.io_instr_MPORT.clk <= clock @[src/main/scala/riscvproc.scala 36:24]
    mem.MPORT.addr <= UInt<4>("h0")
    mem.MPORT.en <= UInt<1>("h1")
    mem.MPORT.clk <= clock
    mem.MPORT.data <= pad(defaultProgram_0, 32)
    mem.MPORT.mask <= UInt<1>("h1")
    mem.MPORT_1.addr <= UInt<4>("h1")
    mem.MPORT_1.en <= UInt<1>("h1")
    mem.MPORT_1.clk <= clock
    mem.MPORT_1.data <= pad(defaultProgram_1, 32)
    mem.MPORT_1.mask <= UInt<1>("h1")
    mem.MPORT_2.addr <= UInt<4>("h2")
    mem.MPORT_2.en <= UInt<1>("h1")
    mem.MPORT_2.clk <= clock
    mem.MPORT_2.data <= pad(defaultProgram_2, 32)
    mem.MPORT_2.mask <= UInt<1>("h1")
    mem.MPORT_3.addr <= UInt<4>("h3")
    mem.MPORT_3.en <= UInt<1>("h1")
    mem.MPORT_3.clk <= clock
    mem.MPORT_3.data <= pad(defaultProgram_3, 32)
    mem.MPORT_3.mask <= UInt<1>("h1")
    mem.MPORT_4.addr <= UInt<4>("h4")
    mem.MPORT_4.en <= UInt<1>("h1")
    mem.MPORT_4.clk <= clock
    mem.MPORT_4.data <= pad(defaultProgram_4, 32)
    mem.MPORT_4.mask <= UInt<1>("h1")
    mem.MPORT_5.addr <= UInt<4>("h5")
    mem.MPORT_5.en <= UInt<1>("h1")
    mem.MPORT_5.clk <= clock
    mem.MPORT_5.data <= pad(defaultProgram_5, 32)
    mem.MPORT_5.mask <= UInt<1>("h1")
    mem.MPORT_6.addr <= UInt<4>("h6")
    mem.MPORT_6.en <= UInt<1>("h1")
    mem.MPORT_6.clk <= clock
    mem.MPORT_6.data <= pad(defaultProgram_6, 32)
    mem.MPORT_6.mask <= UInt<1>("h1")
    mem.MPORT_7.addr <= UInt<4>("h7")
    mem.MPORT_7.en <= UInt<1>("h1")
    mem.MPORT_7.clk <= clock
    mem.MPORT_7.data <= pad(defaultProgram_7, 32)
    mem.MPORT_7.mask <= UInt<1>("h1")
    mem.MPORT_8.addr <= UInt<4>("h8")
    mem.MPORT_8.en <= UInt<1>("h1")
    mem.MPORT_8.clk <= clock
    mem.MPORT_8.data <= pad(defaultProgram_8, 32)
    mem.MPORT_8.mask <= UInt<1>("h1")
    mem.MPORT_9.addr <= UInt<4>("h9")
    mem.MPORT_9.en <= UInt<1>("h1")
    mem.MPORT_9.clk <= clock
    mem.MPORT_9.data <= pad(defaultProgram_9, 32)
    mem.MPORT_9.mask <= UInt<1>("h1")
    mem.MPORT_10.addr <= UInt<4>("ha")
    mem.MPORT_10.en <= UInt<1>("h1")
    mem.MPORT_10.clk <= clock
    mem.MPORT_10.data <= pad(defaultProgram_10, 32)
    mem.MPORT_10.mask <= UInt<1>("h1")
    mem.MPORT_11.addr <= UInt<4>("hb")
    mem.MPORT_11.en <= UInt<1>("h1")
    mem.MPORT_11.clk <= clock
    mem.MPORT_11.data <= pad(defaultProgram_11, 32)
    mem.MPORT_11.mask <= UInt<1>("h1")
    mem.MPORT_12.addr <= UInt<4>("hc")
    mem.MPORT_12.en <= UInt<1>("h1")
    mem.MPORT_12.clk <= clock
    mem.MPORT_12.data <= pad(defaultProgram_12, 32)
    mem.MPORT_12.mask <= UInt<1>("h1")
    mem.MPORT_13.addr <= UInt<4>("hd")
    mem.MPORT_13.en <= UInt<1>("h1")
    mem.MPORT_13.clk <= clock
    mem.MPORT_13.data <= pad(defaultProgram_13, 32)
    mem.MPORT_13.mask <= UInt<1>("h1")
    mem.MPORT_14.addr <= UInt<4>("he")
    mem.MPORT_14.en <= UInt<1>("h1")
    mem.MPORT_14.clk <= clock
    mem.MPORT_14.data <= pad(defaultProgram_14, 32)
    mem.MPORT_14.mask <= UInt<1>("h1")
    mem.MPORT_15.addr <= UInt<4>("hf")
    mem.MPORT_15.en <= UInt<1>("h1")
    mem.MPORT_15.clk <= clock
    mem.MPORT_15.data <= pad(defaultProgram_15, 32)
    mem.MPORT_15.mask <= UInt<1>("h1")

  module Controller :
    input clock : Clock
    input reset : UInt<1>
    input io_opcode : UInt<7> @[src/main/scala/riscvproc.scala 147:20]
    output io_branch : UInt<1> @[src/main/scala/riscvproc.scala 147:20]
    output io_alu_src : UInt<1> @[src/main/scala/riscvproc.scala 147:20]
    output io_regwrite : UInt<1> @[src/main/scala/riscvproc.scala 147:20]
    output io_memread : UInt<1> @[src/main/scala/riscvproc.scala 147:20]
    output io_memwrite : UInt<1> @[src/main/scala/riscvproc.scala 147:20]
    output io_memtoreg : UInt<1> @[src/main/scala/riscvproc.scala 147:20]
    output io_alu_op : UInt<2> @[src/main/scala/riscvproc.scala 147:20]

    node _T = eq(UInt<2>("h3"), io_opcode) @[src/main/scala/riscvproc.scala 167:27]
    node _T_1 = eq(UInt<5>("h13"), io_opcode) @[src/main/scala/riscvproc.scala 167:27]
    node _T_2 = eq(UInt<6>("h23"), io_opcode) @[src/main/scala/riscvproc.scala 167:27]
    node _T_3 = eq(UInt<6>("h33"), io_opcode) @[src/main/scala/riscvproc.scala 167:27]
    node _T_4 = eq(UInt<7>("h63"), io_opcode) @[src/main/scala/riscvproc.scala 167:27]
    node _GEN_0 = mux(_T_4, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/riscvproc.scala 167:27 196:19 164:21]
    node _GEN_1 = mux(_T_4, UInt<2>("h1"), UInt<2>("h0")) @[src/main/scala/riscvproc.scala 167:27 197:19 165:21]
    node _GEN_2 = mux(_T_3, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/riscvproc.scala 161:21 167:27 190:21]
    node _GEN_3 = mux(_T_3, UInt<2>("h2"), _GEN_1) @[src/main/scala/riscvproc.scala 167:27 191:21]
    node _GEN_4 = mux(_T_3, UInt<1>("h0"), _GEN_0) @[src/main/scala/riscvproc.scala 164:21 167:27]
    node _GEN_5 = mux(_T_2, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/riscvproc.scala 159:21 167:27 184:21]
    node _GEN_6 = mux(_T_2, UInt<1>("h0"), _GEN_2) @[src/main/scala/riscvproc.scala 161:21 167:27]
    node _GEN_7 = mux(_T_2, UInt<2>("h0"), _GEN_3) @[src/main/scala/riscvproc.scala 165:21 167:27]
    node _GEN_8 = mux(_T_2, UInt<1>("h0"), _GEN_4) @[src/main/scala/riscvproc.scala 164:21 167:27]
    node _GEN_9 = mux(_T_1, UInt<1>("h1"), _GEN_5) @[src/main/scala/riscvproc.scala 167:27 178:21]
    node _GEN_10 = mux(_T_1, UInt<1>("h1"), _GEN_6) @[src/main/scala/riscvproc.scala 167:27 179:21]
    node _GEN_11 = mux(_T_1, UInt<1>("h0"), _GEN_5) @[src/main/scala/riscvproc.scala 163:21 167:27]
    node _GEN_12 = mux(_T_1, UInt<2>("h0"), _GEN_7) @[src/main/scala/riscvproc.scala 165:21 167:27]
    node _GEN_13 = mux(_T_1, UInt<1>("h0"), _GEN_8) @[src/main/scala/riscvproc.scala 164:21 167:27]
    node _GEN_14 = mux(_T, UInt<1>("h1"), _GEN_9) @[src/main/scala/riscvproc.scala 167:27 170:21]
    node _GEN_15 = mux(_T, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/riscvproc.scala 160:21 167:27 171:21]
    node _GEN_16 = mux(_T, UInt<1>("h1"), _GEN_10) @[src/main/scala/riscvproc.scala 167:27 172:21]
    node _GEN_17 = mux(_T, UInt<1>("h0"), _GEN_11) @[src/main/scala/riscvproc.scala 163:21 167:27]
    node _GEN_18 = mux(_T, UInt<2>("h0"), _GEN_12) @[src/main/scala/riscvproc.scala 165:21 167:27]
    node _GEN_19 = mux(_T, UInt<1>("h0"), _GEN_13) @[src/main/scala/riscvproc.scala 164:21 167:27]
    io_branch <= _GEN_19
    io_alu_src <= _GEN_14
    io_regwrite <= _GEN_16
    io_memread <= _GEN_15
    io_memwrite <= _GEN_17
    io_memtoreg <= _GEN_15
    io_alu_op <= _GEN_18

  module Registers :
    input clock : Clock
    input reset : UInt<1>
    input io_reg_addr1 : UInt<5> @[src/main/scala/riscvproc.scala 60:20]
    input io_reg_addr2 : UInt<5> @[src/main/scala/riscvproc.scala 60:20]
    input io_wr_addr : UInt<5> @[src/main/scala/riscvproc.scala 60:20]
    input io_wr_data : UInt<32> @[src/main/scala/riscvproc.scala 60:20]
    input io_wr_en : UInt<1> @[src/main/scala/riscvproc.scala 60:20]
    output io_reg_data1 : UInt<32> @[src/main/scala/riscvproc.scala 60:20]
    output io_reg_data2 : UInt<32> @[src/main/scala/riscvproc.scala 60:20]
    output io_x1 : UInt<32> @[src/main/scala/riscvproc.scala 60:20]
    output io_x2 : UInt<32> @[src/main/scala/riscvproc.scala 60:20]
    output io_x3 : UInt<32> @[src/main/scala/riscvproc.scala 60:20]

    reg regfile_0 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_0) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_1 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_1) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_2 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_2) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_3 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_3) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_4 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_4) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_5 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_5) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_6 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_6) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_7 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_7) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_8 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_8) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_9 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_9) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_10 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_10) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_11 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_11) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_12 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_12) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_13 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_13) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_14 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_14) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_15 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_15) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_16 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_16) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_17 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_17) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_18 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_18) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_19 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_19) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_20 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_20) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_21 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_21) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_22 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_22) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_23 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_23) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_24 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_24) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_25 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_25) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_26 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_26) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_27 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_27) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_28 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_28) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_29 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_29) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_30 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_30) @[src/main/scala/riscvproc.scala 76:30]
    reg regfile_31 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), regfile_31) @[src/main/scala/riscvproc.scala 76:30]
    node _T = neq(io_wr_addr, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 78:37]
    node _T_1 = and(io_wr_en, _T) @[src/main/scala/riscvproc.scala 78:23]
    node _regfile_io_wr_addr = io_wr_data @[src/main/scala/riscvproc.scala 79:{29,29}]
    node _GEN_0 = mux(eq(UInt<1>("h0"), io_wr_addr), _regfile_io_wr_addr, regfile_0) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_1 = mux(eq(UInt<1>("h1"), io_wr_addr), _regfile_io_wr_addr, regfile_1) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_2 = mux(eq(UInt<2>("h2"), io_wr_addr), _regfile_io_wr_addr, regfile_2) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_3 = mux(eq(UInt<2>("h3"), io_wr_addr), _regfile_io_wr_addr, regfile_3) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_4 = mux(eq(UInt<3>("h4"), io_wr_addr), _regfile_io_wr_addr, regfile_4) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_5 = mux(eq(UInt<3>("h5"), io_wr_addr), _regfile_io_wr_addr, regfile_5) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_6 = mux(eq(UInt<3>("h6"), io_wr_addr), _regfile_io_wr_addr, regfile_6) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_7 = mux(eq(UInt<3>("h7"), io_wr_addr), _regfile_io_wr_addr, regfile_7) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_8 = mux(eq(UInt<4>("h8"), io_wr_addr), _regfile_io_wr_addr, regfile_8) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_9 = mux(eq(UInt<4>("h9"), io_wr_addr), _regfile_io_wr_addr, regfile_9) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_10 = mux(eq(UInt<4>("ha"), io_wr_addr), _regfile_io_wr_addr, regfile_10) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_11 = mux(eq(UInt<4>("hb"), io_wr_addr), _regfile_io_wr_addr, regfile_11) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_12 = mux(eq(UInt<4>("hc"), io_wr_addr), _regfile_io_wr_addr, regfile_12) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_13 = mux(eq(UInt<4>("hd"), io_wr_addr), _regfile_io_wr_addr, regfile_13) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_14 = mux(eq(UInt<4>("he"), io_wr_addr), _regfile_io_wr_addr, regfile_14) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_15 = mux(eq(UInt<4>("hf"), io_wr_addr), _regfile_io_wr_addr, regfile_15) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_16 = mux(eq(UInt<5>("h10"), io_wr_addr), _regfile_io_wr_addr, regfile_16) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_17 = mux(eq(UInt<5>("h11"), io_wr_addr), _regfile_io_wr_addr, regfile_17) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_18 = mux(eq(UInt<5>("h12"), io_wr_addr), _regfile_io_wr_addr, regfile_18) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_19 = mux(eq(UInt<5>("h13"), io_wr_addr), _regfile_io_wr_addr, regfile_19) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_20 = mux(eq(UInt<5>("h14"), io_wr_addr), _regfile_io_wr_addr, regfile_20) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_21 = mux(eq(UInt<5>("h15"), io_wr_addr), _regfile_io_wr_addr, regfile_21) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_22 = mux(eq(UInt<5>("h16"), io_wr_addr), _regfile_io_wr_addr, regfile_22) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_23 = mux(eq(UInt<5>("h17"), io_wr_addr), _regfile_io_wr_addr, regfile_23) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_24 = mux(eq(UInt<5>("h18"), io_wr_addr), _regfile_io_wr_addr, regfile_24) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_25 = mux(eq(UInt<5>("h19"), io_wr_addr), _regfile_io_wr_addr, regfile_25) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_26 = mux(eq(UInt<5>("h1a"), io_wr_addr), _regfile_io_wr_addr, regfile_26) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_27 = mux(eq(UInt<5>("h1b"), io_wr_addr), _regfile_io_wr_addr, regfile_27) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_28 = mux(eq(UInt<5>("h1c"), io_wr_addr), _regfile_io_wr_addr, regfile_28) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_29 = mux(eq(UInt<5>("h1d"), io_wr_addr), _regfile_io_wr_addr, regfile_29) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_30 = mux(eq(UInt<5>("h1e"), io_wr_addr), _regfile_io_wr_addr, regfile_30) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_31 = mux(eq(UInt<5>("h1f"), io_wr_addr), _regfile_io_wr_addr, regfile_31) @[src/main/scala/riscvproc.scala 79:{29,29} 76:30]
    node _GEN_32 = mux(_T_1, _GEN_0, regfile_0) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_33 = mux(_T_1, _GEN_1, regfile_1) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_34 = mux(_T_1, _GEN_2, regfile_2) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_35 = mux(_T_1, _GEN_3, regfile_3) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_36 = mux(_T_1, _GEN_4, regfile_4) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_37 = mux(_T_1, _GEN_5, regfile_5) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_38 = mux(_T_1, _GEN_6, regfile_6) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_39 = mux(_T_1, _GEN_7, regfile_7) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_40 = mux(_T_1, _GEN_8, regfile_8) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_41 = mux(_T_1, _GEN_9, regfile_9) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_42 = mux(_T_1, _GEN_10, regfile_10) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_43 = mux(_T_1, _GEN_11, regfile_11) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_44 = mux(_T_1, _GEN_12, regfile_12) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_45 = mux(_T_1, _GEN_13, regfile_13) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_46 = mux(_T_1, _GEN_14, regfile_14) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_47 = mux(_T_1, _GEN_15, regfile_15) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_48 = mux(_T_1, _GEN_16, regfile_16) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_49 = mux(_T_1, _GEN_17, regfile_17) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_50 = mux(_T_1, _GEN_18, regfile_18) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_51 = mux(_T_1, _GEN_19, regfile_19) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_52 = mux(_T_1, _GEN_20, regfile_20) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_53 = mux(_T_1, _GEN_21, regfile_21) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_54 = mux(_T_1, _GEN_22, regfile_22) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_55 = mux(_T_1, _GEN_23, regfile_23) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_56 = mux(_T_1, _GEN_24, regfile_24) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_57 = mux(_T_1, _GEN_25, regfile_25) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_58 = mux(_T_1, _GEN_26, regfile_26) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_59 = mux(_T_1, _GEN_27, regfile_27) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_60 = mux(_T_1, _GEN_28, regfile_28) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_61 = mux(_T_1, _GEN_29, regfile_29) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_62 = mux(_T_1, _GEN_30, regfile_30) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _GEN_63 = mux(_T_1, _GEN_31, regfile_31) @[src/main/scala/riscvproc.scala 76:30 78:46]
    node _io_reg_data1_T = eq(io_reg_addr1, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 81:42]
    node _GEN_64 = validif(eq(UInt<1>("h0"), io_reg_addr1), regfile_0) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_65 = mux(eq(UInt<1>("h1"), io_reg_addr1), regfile_1, _GEN_64) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_66 = mux(eq(UInt<2>("h2"), io_reg_addr1), regfile_2, _GEN_65) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_67 = mux(eq(UInt<2>("h3"), io_reg_addr1), regfile_3, _GEN_66) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_68 = mux(eq(UInt<3>("h4"), io_reg_addr1), regfile_4, _GEN_67) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_69 = mux(eq(UInt<3>("h5"), io_reg_addr1), regfile_5, _GEN_68) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_70 = mux(eq(UInt<3>("h6"), io_reg_addr1), regfile_6, _GEN_69) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_71 = mux(eq(UInt<3>("h7"), io_reg_addr1), regfile_7, _GEN_70) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_72 = mux(eq(UInt<4>("h8"), io_reg_addr1), regfile_8, _GEN_71) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_73 = mux(eq(UInt<4>("h9"), io_reg_addr1), regfile_9, _GEN_72) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_74 = mux(eq(UInt<4>("ha"), io_reg_addr1), regfile_10, _GEN_73) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_75 = mux(eq(UInt<4>("hb"), io_reg_addr1), regfile_11, _GEN_74) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_76 = mux(eq(UInt<4>("hc"), io_reg_addr1), regfile_12, _GEN_75) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_77 = mux(eq(UInt<4>("hd"), io_reg_addr1), regfile_13, _GEN_76) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_78 = mux(eq(UInt<4>("he"), io_reg_addr1), regfile_14, _GEN_77) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_79 = mux(eq(UInt<4>("hf"), io_reg_addr1), regfile_15, _GEN_78) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_80 = mux(eq(UInt<5>("h10"), io_reg_addr1), regfile_16, _GEN_79) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_81 = mux(eq(UInt<5>("h11"), io_reg_addr1), regfile_17, _GEN_80) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_82 = mux(eq(UInt<5>("h12"), io_reg_addr1), regfile_18, _GEN_81) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_83 = mux(eq(UInt<5>("h13"), io_reg_addr1), regfile_19, _GEN_82) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_84 = mux(eq(UInt<5>("h14"), io_reg_addr1), regfile_20, _GEN_83) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_85 = mux(eq(UInt<5>("h15"), io_reg_addr1), regfile_21, _GEN_84) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_86 = mux(eq(UInt<5>("h16"), io_reg_addr1), regfile_22, _GEN_85) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_87 = mux(eq(UInt<5>("h17"), io_reg_addr1), regfile_23, _GEN_86) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_88 = mux(eq(UInt<5>("h18"), io_reg_addr1), regfile_24, _GEN_87) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_89 = mux(eq(UInt<5>("h19"), io_reg_addr1), regfile_25, _GEN_88) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_90 = mux(eq(UInt<5>("h1a"), io_reg_addr1), regfile_26, _GEN_89) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_91 = mux(eq(UInt<5>("h1b"), io_reg_addr1), regfile_27, _GEN_90) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_92 = mux(eq(UInt<5>("h1c"), io_reg_addr1), regfile_28, _GEN_91) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_93 = mux(eq(UInt<5>("h1d"), io_reg_addr1), regfile_29, _GEN_92) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_94 = mux(eq(UInt<5>("h1e"), io_reg_addr1), regfile_30, _GEN_93) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _GEN_95 = mux(eq(UInt<5>("h1f"), io_reg_addr1), regfile_31, _GEN_94) @[src/main/scala/riscvproc.scala 81:{28,28}]
    node _regfile_io_reg_addr1 = _GEN_95 @[src/main/scala/riscvproc.scala 81:28]
    node _io_reg_data1_T_1 = mux(_io_reg_data1_T, UInt<1>("h0"), _regfile_io_reg_addr1) @[src/main/scala/riscvproc.scala 81:28]
    node _io_reg_data2_T = eq(io_reg_addr2, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 82:42]
    node _GEN_96 = validif(eq(UInt<1>("h0"), io_reg_addr2), regfile_0) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_97 = mux(eq(UInt<1>("h1"), io_reg_addr2), regfile_1, _GEN_96) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_98 = mux(eq(UInt<2>("h2"), io_reg_addr2), regfile_2, _GEN_97) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_99 = mux(eq(UInt<2>("h3"), io_reg_addr2), regfile_3, _GEN_98) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_100 = mux(eq(UInt<3>("h4"), io_reg_addr2), regfile_4, _GEN_99) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_101 = mux(eq(UInt<3>("h5"), io_reg_addr2), regfile_5, _GEN_100) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_102 = mux(eq(UInt<3>("h6"), io_reg_addr2), regfile_6, _GEN_101) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_103 = mux(eq(UInt<3>("h7"), io_reg_addr2), regfile_7, _GEN_102) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_104 = mux(eq(UInt<4>("h8"), io_reg_addr2), regfile_8, _GEN_103) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_105 = mux(eq(UInt<4>("h9"), io_reg_addr2), regfile_9, _GEN_104) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_106 = mux(eq(UInt<4>("ha"), io_reg_addr2), regfile_10, _GEN_105) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_107 = mux(eq(UInt<4>("hb"), io_reg_addr2), regfile_11, _GEN_106) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_108 = mux(eq(UInt<4>("hc"), io_reg_addr2), regfile_12, _GEN_107) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_109 = mux(eq(UInt<4>("hd"), io_reg_addr2), regfile_13, _GEN_108) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_110 = mux(eq(UInt<4>("he"), io_reg_addr2), regfile_14, _GEN_109) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_111 = mux(eq(UInt<4>("hf"), io_reg_addr2), regfile_15, _GEN_110) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_112 = mux(eq(UInt<5>("h10"), io_reg_addr2), regfile_16, _GEN_111) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_113 = mux(eq(UInt<5>("h11"), io_reg_addr2), regfile_17, _GEN_112) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_114 = mux(eq(UInt<5>("h12"), io_reg_addr2), regfile_18, _GEN_113) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_115 = mux(eq(UInt<5>("h13"), io_reg_addr2), regfile_19, _GEN_114) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_116 = mux(eq(UInt<5>("h14"), io_reg_addr2), regfile_20, _GEN_115) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_117 = mux(eq(UInt<5>("h15"), io_reg_addr2), regfile_21, _GEN_116) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_118 = mux(eq(UInt<5>("h16"), io_reg_addr2), regfile_22, _GEN_117) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_119 = mux(eq(UInt<5>("h17"), io_reg_addr2), regfile_23, _GEN_118) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_120 = mux(eq(UInt<5>("h18"), io_reg_addr2), regfile_24, _GEN_119) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_121 = mux(eq(UInt<5>("h19"), io_reg_addr2), regfile_25, _GEN_120) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_122 = mux(eq(UInt<5>("h1a"), io_reg_addr2), regfile_26, _GEN_121) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_123 = mux(eq(UInt<5>("h1b"), io_reg_addr2), regfile_27, _GEN_122) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_124 = mux(eq(UInt<5>("h1c"), io_reg_addr2), regfile_28, _GEN_123) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_125 = mux(eq(UInt<5>("h1d"), io_reg_addr2), regfile_29, _GEN_124) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_126 = mux(eq(UInt<5>("h1e"), io_reg_addr2), regfile_30, _GEN_125) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _GEN_127 = mux(eq(UInt<5>("h1f"), io_reg_addr2), regfile_31, _GEN_126) @[src/main/scala/riscvproc.scala 82:{28,28}]
    node _regfile_io_reg_addr2 = _GEN_127 @[src/main/scala/riscvproc.scala 82:28]
    node _io_reg_data2_T_1 = mux(_io_reg_data2_T, UInt<1>("h0"), _regfile_io_reg_addr2) @[src/main/scala/riscvproc.scala 82:28]
    node _regfile_WIRE_0 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_1 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_2 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_3 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_4 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_5 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_6 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_7 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_8 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_9 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_10 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_11 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_12 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_13 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_14 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_15 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_16 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_17 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_18 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_19 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_20 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_21 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_22 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_23 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_24 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_25 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_26 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_27 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_28 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_29 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_30 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    node _regfile_WIRE_31 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 76:{38,38}]
    io_reg_data1 <= _io_reg_data1_T_1 @[src/main/scala/riscvproc.scala 81:22]
    io_reg_data2 <= _io_reg_data2_T_1 @[src/main/scala/riscvproc.scala 82:22]
    io_x1 <= regfile_1 @[src/main/scala/riscvproc.scala 85:15]
    io_x2 <= regfile_2 @[src/main/scala/riscvproc.scala 86:15]
    io_x3 <= regfile_3 @[src/main/scala/riscvproc.scala 87:15]
    regfile_0 <= mux(reset, _regfile_WIRE_0, _GEN_32) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_1 <= mux(reset, _regfile_WIRE_1, _GEN_33) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_2 <= mux(reset, _regfile_WIRE_2, _GEN_34) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_3 <= mux(reset, _regfile_WIRE_3, _GEN_35) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_4 <= mux(reset, _regfile_WIRE_4, _GEN_36) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_5 <= mux(reset, _regfile_WIRE_5, _GEN_37) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_6 <= mux(reset, _regfile_WIRE_6, _GEN_38) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_7 <= mux(reset, _regfile_WIRE_7, _GEN_39) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_8 <= mux(reset, _regfile_WIRE_8, _GEN_40) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_9 <= mux(reset, _regfile_WIRE_9, _GEN_41) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_10 <= mux(reset, _regfile_WIRE_10, _GEN_42) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_11 <= mux(reset, _regfile_WIRE_11, _GEN_43) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_12 <= mux(reset, _regfile_WIRE_12, _GEN_44) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_13 <= mux(reset, _regfile_WIRE_13, _GEN_45) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_14 <= mux(reset, _regfile_WIRE_14, _GEN_46) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_15 <= mux(reset, _regfile_WIRE_15, _GEN_47) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_16 <= mux(reset, _regfile_WIRE_16, _GEN_48) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_17 <= mux(reset, _regfile_WIRE_17, _GEN_49) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_18 <= mux(reset, _regfile_WIRE_18, _GEN_50) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_19 <= mux(reset, _regfile_WIRE_19, _GEN_51) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_20 <= mux(reset, _regfile_WIRE_20, _GEN_52) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_21 <= mux(reset, _regfile_WIRE_21, _GEN_53) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_22 <= mux(reset, _regfile_WIRE_22, _GEN_54) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_23 <= mux(reset, _regfile_WIRE_23, _GEN_55) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_24 <= mux(reset, _regfile_WIRE_24, _GEN_56) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_25 <= mux(reset, _regfile_WIRE_25, _GEN_57) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_26 <= mux(reset, _regfile_WIRE_26, _GEN_58) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_27 <= mux(reset, _regfile_WIRE_27, _GEN_59) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_28 <= mux(reset, _regfile_WIRE_28, _GEN_60) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_29 <= mux(reset, _regfile_WIRE_29, _GEN_61) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_30 <= mux(reset, _regfile_WIRE_30, _GEN_62) @[src/main/scala/riscvproc.scala 76:{30,30}]
    regfile_31 <= mux(reset, _regfile_WIRE_31, _GEN_63) @[src/main/scala/riscvproc.scala 76:{30,30}]

  module ImmGen :
    input clock : Clock
    input reset : UInt<1>
    input io_instr : UInt<32> @[src/main/scala/riscvproc.scala 91:20]
    output io_imm_val : UInt<32> @[src/main/scala/riscvproc.scala 91:20]

    node op_code = bits(io_instr, 6, 0) @[src/main/scala/riscvproc.scala 96:31]
    node _T = eq(UInt<2>("h3"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _T_1 = eq(UInt<5>("h13"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _T_2 = or(_T, _T_1) @[src/main/scala/riscvproc.scala 100:25]
    node _io_imm_val_T = bits(io_instr, 31, 31) @[src/main/scala/riscvproc.scala 103:44]
    node _io_imm_val_T_1 = mux(_io_imm_val_T, UInt<20>("hfffff"), UInt<20>("h0")) @[src/main/scala/riscvproc.scala 103:31]
    node _io_imm_val_T_2 = bits(io_instr, 31, 20) @[src/main/scala/riscvproc.scala 103:59]
    node _io_imm_val_T_3 = cat(_io_imm_val_T_1, _io_imm_val_T_2) @[src/main/scala/riscvproc.scala 103:26]
    node _T_3 = eq(UInt<6>("h23"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _io_imm_val_T_4 = bits(io_instr, 31, 31) @[src/main/scala/riscvproc.scala 108:44]
    node _io_imm_val_T_5 = mux(_io_imm_val_T_4, UInt<20>("hfffff"), UInt<20>("h0")) @[src/main/scala/riscvproc.scala 108:31]
    node _io_imm_val_T_6 = bits(io_instr, 31, 25) @[src/main/scala/riscvproc.scala 108:59]
    node _io_imm_val_T_7 = bits(io_instr, 11, 7) @[src/main/scala/riscvproc.scala 108:77]
    node io_imm_val_hi = cat(_io_imm_val_T_5, _io_imm_val_T_6) @[src/main/scala/riscvproc.scala 108:26]
    node _io_imm_val_T_8 = cat(io_imm_val_hi, _io_imm_val_T_7) @[src/main/scala/riscvproc.scala 108:26]
    node _T_4 = eq(UInt<7>("h63"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _io_imm_val_T_9 = bits(io_instr, 31, 31) @[src/main/scala/riscvproc.scala 114:30]
    node _io_imm_val_T_10 = mux(_io_imm_val_T_9, UInt<20>("hfffff"), UInt<20>("h0")) @[src/main/scala/riscvproc.scala 114:17]
    node _io_imm_val_T_11 = bits(io_instr, 7, 7) @[src/main/scala/riscvproc.scala 115:21]
    node _io_imm_val_T_12 = bits(io_instr, 30, 25) @[src/main/scala/riscvproc.scala 116:21]
    node _io_imm_val_T_13 = bits(io_instr, 11, 8) @[src/main/scala/riscvproc.scala 117:21]
    node io_imm_val_lo = cat(_io_imm_val_T_13, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 113:26]
    node io_imm_val_hi_hi = cat(_io_imm_val_T_10, _io_imm_val_T_11) @[src/main/scala/riscvproc.scala 113:26]
    node io_imm_val_hi_1 = cat(io_imm_val_hi_hi, _io_imm_val_T_12) @[src/main/scala/riscvproc.scala 113:26]
    node _io_imm_val_T_14 = cat(io_imm_val_hi_1, io_imm_val_lo) @[src/main/scala/riscvproc.scala 113:26]
    node _T_5 = eq(UInt<6>("h37"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _T_6 = eq(UInt<5>("h17"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _T_7 = or(_T_5, _T_6) @[src/main/scala/riscvproc.scala 100:25]
    node _io_imm_val_T_15 = bits(io_instr, 31, 12) @[src/main/scala/riscvproc.scala 124:35]
    node _io_imm_val_T_16 = cat(_io_imm_val_T_15, UInt<12>("h0")) @[src/main/scala/riscvproc.scala 124:26]
    node _T_8 = eq(UInt<7>("h6f"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _io_imm_val_T_17 = bits(io_instr, 31, 31) @[src/main/scala/riscvproc.scala 130:30]
    node _io_imm_val_T_18 = mux(_io_imm_val_T_17, UInt<11>("h7ff"), UInt<11>("h0")) @[src/main/scala/riscvproc.scala 130:17]
    node _io_imm_val_T_19 = bits(io_instr, 31, 31) @[src/main/scala/riscvproc.scala 131:21]
    node _io_imm_val_T_20 = bits(io_instr, 19, 12) @[src/main/scala/riscvproc.scala 132:21]
    node _io_imm_val_T_21 = bits(io_instr, 20, 20) @[src/main/scala/riscvproc.scala 133:21]
    node _io_imm_val_T_22 = bits(io_instr, 30, 21) @[src/main/scala/riscvproc.scala 134:21]
    node io_imm_val_lo_hi = cat(_io_imm_val_T_21, _io_imm_val_T_22) @[src/main/scala/riscvproc.scala 129:26]
    node io_imm_val_lo_1 = cat(io_imm_val_lo_hi, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 129:26]
    node io_imm_val_hi_hi_1 = cat(_io_imm_val_T_18, _io_imm_val_T_19) @[src/main/scala/riscvproc.scala 129:26]
    node io_imm_val_hi_2 = cat(io_imm_val_hi_hi_1, _io_imm_val_T_20) @[src/main/scala/riscvproc.scala 129:26]
    node _io_imm_val_T_23 = cat(io_imm_val_hi_2, io_imm_val_lo_1) @[src/main/scala/riscvproc.scala 129:26]
    node _T_9 = eq(UInt<7>("h67"), op_code) @[src/main/scala/riscvproc.scala 100:25]
    node _io_imm_val_T_24 = bits(io_instr, 31, 31) @[src/main/scala/riscvproc.scala 141:44]
    node _io_imm_val_T_25 = mux(_io_imm_val_T_24, UInt<20>("hfffff"), UInt<20>("h0")) @[src/main/scala/riscvproc.scala 141:31]
    node _io_imm_val_T_26 = bits(io_instr, 31, 20) @[src/main/scala/riscvproc.scala 141:59]
    node _io_imm_val_T_27 = cat(_io_imm_val_T_25, _io_imm_val_T_26) @[src/main/scala/riscvproc.scala 141:26]
    node _GEN_0 = mux(_T_9, _io_imm_val_T_27, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 100:25 141:20 98:20]
    node _GEN_1 = mux(_T_8, _io_imm_val_T_23, _GEN_0) @[src/main/scala/riscvproc.scala 100:25 129:20]
    node _GEN_2 = mux(_T_7, _io_imm_val_T_16, _GEN_1) @[src/main/scala/riscvproc.scala 100:25 124:20]
    node _GEN_3 = mux(_T_4, _io_imm_val_T_14, _GEN_2) @[src/main/scala/riscvproc.scala 100:25 113:20]
    node _GEN_4 = mux(_T_3, _io_imm_val_T_8, _GEN_3) @[src/main/scala/riscvproc.scala 100:25 108:20]
    node _GEN_5 = mux(_T_2, _io_imm_val_T_3, _GEN_4) @[src/main/scala/riscvproc.scala 100:25 103:20]
    io_imm_val <= _GEN_5

  module ALUCtrl :
    input clock : Clock
    input reset : UInt<1>
    input io_alu_op : UInt<2> @[src/main/scala/riscvproc.scala 274:20]
    input io_funct73 : UInt<4> @[src/main/scala/riscvproc.scala 274:20]
    output io_operation : UInt<4> @[src/main/scala/riscvproc.scala 274:20]

    node _T = eq(UInt<1>("h0"), io_alu_op) @[src/main/scala/riscvproc.scala 283:27]
    node _T_1 = eq(UInt<1>("h1"), io_alu_op) @[src/main/scala/riscvproc.scala 283:27]
    node _T_2 = eq(UInt<2>("h2"), io_alu_op) @[src/main/scala/riscvproc.scala 283:27]
    node _T_3 = eq(UInt<4>("h0"), io_funct73) @[src/main/scala/riscvproc.scala 291:28]
    node _T_4 = eq(UInt<4>("h8"), io_funct73) @[src/main/scala/riscvproc.scala 291:28]
    node _T_5 = eq(UInt<4>("h7"), io_funct73) @[src/main/scala/riscvproc.scala 291:28]
    node _T_6 = eq(UInt<4>("h6"), io_funct73) @[src/main/scala/riscvproc.scala 291:28]
    node _GEN_0 = mux(_T_6, UInt<4>("h1"), UInt<4>("h2")) @[src/main/scala/riscvproc.scala 281:22 291:28 302:26]
    node _GEN_1 = mux(_T_5, UInt<4>("h0"), _GEN_0) @[src/main/scala/riscvproc.scala 291:28 299:26]
    node _GEN_2 = mux(_T_4, UInt<4>("h6"), _GEN_1) @[src/main/scala/riscvproc.scala 291:28 296:26]
    node _GEN_3 = mux(_T_3, UInt<4>("h2"), _GEN_2) @[src/main/scala/riscvproc.scala 291:28 293:26]
    node _GEN_4 = mux(_T_2, _GEN_3, UInt<4>("h2")) @[src/main/scala/riscvproc.scala 281:22 283:27]
    node _GEN_5 = mux(_T_1, UInt<4>("h6"), _GEN_4) @[src/main/scala/riscvproc.scala 283:27 288:22]
    node _GEN_6 = mux(_T, UInt<4>("h2"), _GEN_5) @[src/main/scala/riscvproc.scala 283:27 285:22]
    io_operation <= _GEN_6

  module ForwardingUnit :
    input clock : Clock
    input reset : UInt<1>
    input io_id_ex_rs1 : UInt<5> @[src/main/scala/riscvproc.scala 204:22]
    input io_id_ex_rs2 : UInt<5> @[src/main/scala/riscvproc.scala 204:22]
    input io_id_ex_mem_read : UInt<1> @[src/main/scala/riscvproc.scala 204:22]
    input io_ex_mem_rd : UInt<5> @[src/main/scala/riscvproc.scala 204:22]
    input io_ex_mem_reg_write : UInt<1> @[src/main/scala/riscvproc.scala 204:22]
    input io_ex_mem_alu_result : UInt<32> @[src/main/scala/riscvproc.scala 204:22]
    input io_mem_wb_rd : UInt<5> @[src/main/scala/riscvproc.scala 204:22]
    input io_mem_wb_reg_write : UInt<1> @[src/main/scala/riscvproc.scala 204:22]
    input io_mem_wb_data : UInt<32> @[src/main/scala/riscvproc.scala 204:22]
    output io_forward_a : UInt<2> @[src/main/scala/riscvproc.scala 204:22]
    output io_forward_b : UInt<2> @[src/main/scala/riscvproc.scala 204:22]
    output io_load_use_hazard : UInt<1> @[src/main/scala/riscvproc.scala 204:22]

    node _T = eq(io_ex_mem_rd, io_id_ex_rs1) @[src/main/scala/riscvproc.scala 229:53]
    node _T_1 = and(io_ex_mem_reg_write, _T) @[src/main/scala/riscvproc.scala 229:36]
    node _T_2 = neq(io_ex_mem_rd, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 229:88]
    node _T_3 = and(_T_1, _T_2) @[src/main/scala/riscvproc.scala 229:71]
    node _T_4 = eq(io_mem_wb_rd, io_id_ex_rs1) @[src/main/scala/riscvproc.scala 231:59]
    node _T_5 = and(io_mem_wb_reg_write, _T_4) @[src/main/scala/riscvproc.scala 231:42]
    node _T_6 = neq(io_mem_wb_rd, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 231:94]
    node _T_7 = and(_T_5, _T_6) @[src/main/scala/riscvproc.scala 231:77]
    node _GEN_0 = mux(_T_7, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/riscvproc.scala 231:104 228:24 232:26]
    node _GEN_1 = mux(_T_3, UInt<2>("h2"), _GEN_0) @[src/main/scala/riscvproc.scala 229:98 230:26]
    node _T_8 = eq(io_ex_mem_rd, io_id_ex_rs2) @[src/main/scala/riscvproc.scala 237:53]
    node _T_9 = and(io_ex_mem_reg_write, _T_8) @[src/main/scala/riscvproc.scala 237:36]
    node _T_10 = neq(io_ex_mem_rd, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 237:88]
    node _T_11 = and(_T_9, _T_10) @[src/main/scala/riscvproc.scala 237:71]
    node _T_12 = eq(io_mem_wb_rd, io_id_ex_rs2) @[src/main/scala/riscvproc.scala 239:59]
    node _T_13 = and(io_mem_wb_reg_write, _T_12) @[src/main/scala/riscvproc.scala 239:42]
    node _T_14 = neq(io_mem_wb_rd, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 239:94]
    node _T_15 = and(_T_13, _T_14) @[src/main/scala/riscvproc.scala 239:77]
    node _GEN_2 = mux(_T_15, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/riscvproc.scala 239:104 236:24 240:26]
    node _GEN_3 = mux(_T_11, UInt<2>("h2"), _GEN_2) @[src/main/scala/riscvproc.scala 237:98 238:26]
    node _io_load_use_hazard_T = eq(io_id_ex_rs1, io_ex_mem_rd) @[src/main/scala/riscvproc.scala 245:39]
    node _io_load_use_hazard_T_1 = eq(io_id_ex_rs2, io_ex_mem_rd) @[src/main/scala/riscvproc.scala 246:38]
    node _io_load_use_hazard_T_2 = or(_io_load_use_hazard_T, _io_load_use_hazard_T_1) @[src/main/scala/riscvproc.scala 245:57]
    node _io_load_use_hazard_T_3 = and(io_id_ex_mem_read, _io_load_use_hazard_T_2) @[src/main/scala/riscvproc.scala 244:51]
    node _io_load_use_hazard_T_4 = neq(io_ex_mem_rd, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 247:38]
    node _io_load_use_hazard_T_5 = and(_io_load_use_hazard_T_3, _io_load_use_hazard_T_4) @[src/main/scala/riscvproc.scala 246:57]
    io_forward_a <= _GEN_1
    io_forward_b <= _GEN_3
    io_load_use_hazard <= _io_load_use_hazard_T_5 @[src/main/scala/riscvproc.scala 244:30]

  module ALU :
    input clock : Clock
    input reset : UInt<1>
    input io_ALUoperation : UInt<4> @[src/main/scala/riscvproc.scala 251:20]
    input io_operand1 : UInt<32> @[src/main/scala/riscvproc.scala 251:20]
    input io_operand2 : UInt<32> @[src/main/scala/riscvproc.scala 251:20]
    output io_ALUresult : UInt<32> @[src/main/scala/riscvproc.scala 251:20]
    output io_zero : UInt<1> @[src/main/scala/riscvproc.scala 251:20]

    node _T = eq(UInt<1>("h0"), io_ALUoperation) @[src/main/scala/riscvproc.scala 261:33]
    node _io_ALUresult_T = and(io_operand1, io_operand2) @[src/main/scala/riscvproc.scala 262:53]
    node _T_1 = eq(UInt<1>("h1"), io_ALUoperation) @[src/main/scala/riscvproc.scala 261:33]
    node _io_ALUresult_T_1 = or(io_operand1, io_operand2) @[src/main/scala/riscvproc.scala 263:53]
    node _T_2 = eq(UInt<2>("h2"), io_ALUoperation) @[src/main/scala/riscvproc.scala 261:33]
    node _io_ALUresult_T_2 = add(io_operand1, io_operand2) @[src/main/scala/riscvproc.scala 264:53]
    node _io_ALUresult_T_3 = tail(_io_ALUresult_T_2, 1) @[src/main/scala/riscvproc.scala 264:53]
    node _T_3 = eq(UInt<3>("h6"), io_ALUoperation) @[src/main/scala/riscvproc.scala 261:33]
    node _io_ALUresult_T_4 = sub(io_operand1, io_operand2) @[src/main/scala/riscvproc.scala 265:53]
    node _io_ALUresult_T_5 = tail(_io_ALUresult_T_4, 1) @[src/main/scala/riscvproc.scala 265:53]
    node _GEN_0 = mux(_T_3, _io_ALUresult_T_5, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 260:22 261:33 265:38]
    node _GEN_1 = mux(_T_2, _io_ALUresult_T_3, _GEN_0) @[src/main/scala/riscvproc.scala 261:33 264:38]
    node _GEN_2 = mux(_T_1, _io_ALUresult_T_1, _GEN_1) @[src/main/scala/riscvproc.scala 261:33 263:38]
    node _GEN_3 = mux(_T, _io_ALUresult_T, _GEN_2) @[src/main/scala/riscvproc.scala 261:33 262:38]
    node _io_zero_T = eq(io_ALUoperation, UInt<4>("h6")) @[src/main/scala/riscvproc.scala 269:37]
    node _io_zero_T_1 = eq(io_ALUresult, UInt<32>("h0")) @[src/main/scala/riscvproc.scala 270:31]
    node _io_zero_T_2 = and(_io_zero_T, _io_zero_T_1) @[src/main/scala/riscvproc.scala 269:57]
    io_ALUresult <= _GEN_3
    io_zero <= _io_zero_T_2 @[src/main/scala/riscvproc.scala 269:17]

  module DataMemory :
    input clock : Clock
    input reset : UInt<1>
    input io_wr_en : UInt<1> @[src/main/scala/riscvproc.scala 40:20]
    input io_rd_en : UInt<1> @[src/main/scala/riscvproc.scala 40:20]
    input io_addr : UInt<32> @[src/main/scala/riscvproc.scala 40:20]
    input io_wr_data : UInt<32> @[src/main/scala/riscvproc.scala 40:20]
    output io_rdata : UInt<32> @[src/main/scala/riscvproc.scala 40:20]

    reg mem_0 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), mem_0) @[src/main/scala/riscvproc.scala 49:26]
    reg mem_1 : UInt<32>, clock with :
      reset => (UInt<1>("h0"), mem_1) @[src/main/scala/riscvproc.scala 49:26]
    node _T = asUInt(reset) @[src/main/scala/riscvproc.scala 51:21]
    node _T_1 = eq(_T, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 51:14]
    node _T_2 = bits(io_addr, 0, 0) @[src/main/scala/riscvproc.scala 53:20]
    node _mem_T_2 = io_wr_data @[src/main/scala/riscvproc.scala 53:{25,25}]
    node _GEN_0 = mux(eq(UInt<1>("h0"), _T_2), _mem_T_2, mem_0) @[src/main/scala/riscvproc.scala 53:{25,25} 49:26]
    node _GEN_1 = mux(eq(UInt<1>("h1"), _T_2), _mem_T_2, mem_1) @[src/main/scala/riscvproc.scala 53:{25,25} 49:26]
    node _GEN_2 = mux(io_wr_en, _GEN_0, mem_0) @[src/main/scala/riscvproc.scala 52:24 49:26]
    node _GEN_3 = mux(io_wr_en, _GEN_1, mem_1) @[src/main/scala/riscvproc.scala 52:24 49:26]
    node _GEN_4 = mux(_T_1, _GEN_2, mem_0) @[src/main/scala/riscvproc.scala 49:26 51:29]
    node _GEN_5 = mux(_T_1, _GEN_3, mem_1) @[src/main/scala/riscvproc.scala 49:26 51:29]
    node _io_rdata_T = bits(io_addr, 0, 0) @[src/main/scala/riscvproc.scala 56:46]
    node _GEN_6 = validif(eq(UInt<1>("h0"), _io_rdata_T), mem_0) @[src/main/scala/riscvproc.scala 56:{24,24}]
    node _GEN_7 = mux(eq(UInt<1>("h1"), _io_rdata_T), mem_1, _GEN_6) @[src/main/scala/riscvproc.scala 56:{24,24}]
    node _mem_io_rdata_T = _GEN_7 @[src/main/scala/riscvproc.scala 56:24]
    node _io_rdata_T_1 = mux(io_rd_en, _mem_io_rdata_T, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 56:24]
    node _mem_WIRE_0 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 49:{34,34}]
    node _mem_WIRE_1 = UInt<32>("h0") @[src/main/scala/riscvproc.scala 49:{34,34}]
    io_rdata <= _io_rdata_T_1 @[src/main/scala/riscvproc.scala 56:18]
    mem_0 <= mux(reset, _mem_WIRE_0, _GEN_4) @[src/main/scala/riscvproc.scala 49:{26,26}]
    mem_1 <= mux(reset, _mem_WIRE_1, _GEN_5) @[src/main/scala/riscvproc.scala 49:{26,26}]

  module RiscV :
    input clock : Clock
    input reset : UInt<1>
    output io_x1 : UInt<32> @[src/main/scala/riscvproc.scala 11:14]
    output io_x2 : UInt<32> @[src/main/scala/riscvproc.scala 11:14]
    output io_x3 : UInt<32> @[src/main/scala/riscvproc.scala 11:14]

    inst instr_mem of InstructionMemory @[src/main/scala/riscvproc.scala 332:31]
    inst ctrl of Controller @[src/main/scala/riscvproc.scala 340:26]
    inst reg_file of Registers @[src/main/scala/riscvproc.scala 358:30]
    inst imm_gen of ImmGen @[src/main/scala/riscvproc.scala 365:29]
    inst alu_ctrl of ALUCtrl @[src/main/scala/riscvproc.scala 370:30]
    inst forwarding of ForwardingUnit @[src/main/scala/riscvproc.scala 375:32]
    inst alu of ALU @[src/main/scala/riscvproc.scala 376:25]
    inst data_mem of DataMemory @[src/main/scala/riscvproc.scala 474:30]
    node rst_n = eq(reset, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 310:21]
    reg pc_current : UInt<32>, clock with :
      reset => (UInt<1>("h0"), pc_current) @[src/main/scala/riscvproc.scala 315:33]
    reg f_pc : UInt<32>, clock with :
      reset => (UInt<1>("h0"), f_pc) @[src/main/scala/riscvproc.scala 321:27]
    node _GEN_0 = mux(rst_n, pc_current, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 322:21 323:14 325:14]
    node _result_T = add(pc_current, UInt<3>("h4")) @[src/main/scala/riscvproc.scala 329:33]
    node result = tail(_result_T, 1) @[src/main/scala/riscvproc.scala 329:33]
    reg if_reg : UInt<32>, clock with :
      reset => (UInt<1>("h0"), if_reg) @[src/main/scala/riscvproc.scala 337:29]
    node _ctrl_io_opcode_T = bits(if_reg, 6, 0) @[src/main/scala/riscvproc.scala 341:33]
    reg waddr : UInt<5>, clock with :
      reset => (UInt<1>("h0"), waddr) @[src/main/scala/riscvproc.scala 351:28]
    reg reg_wr : UInt<1>, clock with :
      reset => (UInt<1>("h0"), reg_wr) @[src/main/scala/riscvproc.scala 352:29]
    reg mem_rd : UInt<1>, clock with :
      reset => (UInt<1>("h0"), mem_rd) @[src/main/scala/riscvproc.scala 353:29]
    reg mem_wr : UInt<1>, clock with :
      reset => (UInt<1>("h0"), mem_wr) @[src/main/scala/riscvproc.scala 354:29]
    reg mr_reg : UInt<1>, clock with :
      reset => (UInt<1>("h0"), mr_reg) @[src/main/scala/riscvproc.scala 355:29]
    node _reg_file_io_reg_addr1_T = bits(if_reg, 19, 15) @[src/main/scala/riscvproc.scala 359:40]
    node _reg_file_io_reg_addr2_T = bits(if_reg, 24, 20) @[src/main/scala/riscvproc.scala 360:40]
    node _alu_ctrl_io_funct73_T = bits(if_reg, 30, 30) @[src/main/scala/riscvproc.scala 372:42]
    node _alu_ctrl_io_funct73_T_1 = bits(if_reg, 14, 12) @[src/main/scala/riscvproc.scala 372:54]
    node _alu_ctrl_io_funct73_T_2 = cat(_alu_ctrl_io_funct73_T, _alu_ctrl_io_funct73_T_1) @[src/main/scala/riscvproc.scala 372:35]
    reg alu_reg : UInt<32>, clock with :
      reset => (UInt<1>("h0"), alu_reg) @[src/main/scala/riscvproc.scala 379:30]
    reg wrd_mem : UInt<32>, clock with :
      reset => (UInt<1>("h0"), wrd_mem) @[src/main/scala/riscvproc.scala 380:30]
    node _forwarding_io_id_ex_rs1_T = bits(if_reg, 19, 15) @[src/main/scala/riscvproc.scala 383:42]
    node _forwarding_io_id_ex_rs2_T = bits(if_reg, 24, 20) @[src/main/scala/riscvproc.scala 384:42]
    node _forward_alu_a_T = eq(forwarding.io_forward_a, UInt<1>("h1")) @[src/main/scala/riscvproc.scala 399:38]
    node _forward_alu_a_T_1 = eq(forwarding.io_forward_a, UInt<2>("h2")) @[src/main/scala/riscvproc.scala 400:38]
    node _forward_alu_a_T_2 = mux(_forward_alu_a_T_1, reg_file.io_wr_data, reg_file.io_reg_data1) @[src/main/scala/chisel3/util/Mux.scala 141:16]
    node forward_alu_a = mux(_forward_alu_a_T, alu_reg, _forward_alu_a_T_2) @[src/main/scala/chisel3/util/Mux.scala 141:16]
    node _forward_alu_b_T = eq(forwarding.io_forward_b, UInt<1>("h1")) @[src/main/scala/riscvproc.scala 407:38]
    node _forward_alu_b_T_1 = eq(forwarding.io_forward_b, UInt<2>("h2")) @[src/main/scala/riscvproc.scala 408:38]
    node _forward_alu_b_T_2 = mux(_forward_alu_b_T_1, reg_file.io_wr_data, reg_file.io_reg_data2) @[src/main/scala/chisel3/util/Mux.scala 141:16]
    node forward_alu_b = mux(_forward_alu_b_T, alu_reg, _forward_alu_b_T_2) @[src/main/scala/chisel3/util/Mux.scala 141:16]
    node _GEN_1 = mux(rst_n, alu.io_ALUresult, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 412:21 413:17 416:17]
    node _GEN_2 = mux(rst_n, forward_alu_b, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 412:21 414:17 417:17]
    node imm_out = mux(ctrl.io_alu_src, imm_gen.io_imm_val, forward_alu_b) @[src/main/scala/riscvproc.scala 423:26]
    node _addresult_T = add(f_pc, imm_gen.io_imm_val) @[src/main/scala/riscvproc.scala 427:30]
    node addresult = tail(_addresult_T, 1) @[src/main/scala/riscvproc.scala 427:30]
    node _pc_next_T = and(ctrl.io_branch, alu.io_zero) @[src/main/scala/riscvproc.scala 438:28]
    node _pc_next_T_1 = mux(_pc_next_T, addresult, result) @[src/main/scala/riscvproc.scala 438:20]
    node _pc_next_T_2 = mux(forwarding.io_load_use_hazard, pc_current, _pc_next_T_1) @[src/main/scala/riscvproc.scala 435:23]
    node _waddr_T = bits(if_reg, 11, 7) @[src/main/scala/riscvproc.scala 455:32]
    node _GEN_3 = mux(forwarding.io_load_use_hazard, UInt<1>("h0"), _waddr_T) @[src/main/scala/riscvproc.scala 444:25 446:23 455:23]
    node _GEN_4 = mux(forwarding.io_load_use_hazard, UInt<1>("h0"), ctrl.io_regwrite) @[src/main/scala/riscvproc.scala 444:25 447:24 456:24]
    node _GEN_5 = mux(forwarding.io_load_use_hazard, UInt<1>("h0"), ctrl.io_memread) @[src/main/scala/riscvproc.scala 444:25 448:24 457:24]
    node _GEN_6 = mux(forwarding.io_load_use_hazard, UInt<1>("h0"), ctrl.io_memwrite) @[src/main/scala/riscvproc.scala 444:25 449:24 458:24]
    node _GEN_7 = mux(forwarding.io_load_use_hazard, UInt<1>("h0"), ctrl.io_memtoreg) @[src/main/scala/riscvproc.scala 444:25 450:24 459:24]
    node _GEN_8 = mux(forwarding.io_load_use_hazard, UInt<5>("h13"), instr_mem.io_instr) @[src/main/scala/riscvproc.scala 444:25 453:24 461:24]
    node _GEN_9 = mux(rst_n, _GEN_3, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 442:21 464:19]
    node _GEN_10 = mux(rst_n, _GEN_4, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 442:21 465:20]
    node _GEN_11 = mux(rst_n, _GEN_5, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 442:21 466:20]
    node _GEN_12 = mux(rst_n, _GEN_6, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 442:21 467:20]
    node _GEN_13 = mux(rst_n, _GEN_7, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 442:21 468:20]
    node _GEN_14 = mux(rst_n, _GEN_8, UInt<1>("h0")) @[src/main/scala/riscvproc.scala 442:21 470:20]
    node _reg_file_io_wr_data_T = mux(mr_reg, data_mem.io_rdata, alu_reg) @[src/main/scala/riscvproc.scala 481:35]
    node pc_next = _pc_next_T_2 @[src/main/scala/riscvproc.scala 316:27 435:17]
    io_x1 <= reg_file.io_x1 @[src/main/scala/riscvproc.scala 484:15]
    io_x2 <= reg_file.io_x2 @[src/main/scala/riscvproc.scala 485:15]
    io_x3 <= reg_file.io_x3 @[src/main/scala/riscvproc.scala 486:15]
    pc_current <= mux(reset, UInt<32>("h0"), pc_next) @[src/main/scala/riscvproc.scala 315:{33,33} 319:20]
    f_pc <= mux(reset, UInt<32>("h0"), _GEN_0) @[src/main/scala/riscvproc.scala 321:{27,27}]
    instr_mem.clock <= clock
    instr_mem.reset <= reset
    instr_mem.io_addr <= pc_current @[src/main/scala/riscvproc.scala 333:27]
    if_reg <= mux(reset, UInt<32>("h0"), _GEN_14) @[src/main/scala/riscvproc.scala 337:{29,29}]
    ctrl.clock <= clock
    ctrl.reset <= reset
    ctrl.io_opcode <= _ctrl_io_opcode_T @[src/main/scala/riscvproc.scala 341:24]
    waddr <= mux(reset, UInt<5>("h0"), _GEN_9) @[src/main/scala/riscvproc.scala 351:{28,28}]
    reg_wr <= mux(reset, UInt<1>("h0"), _GEN_10) @[src/main/scala/riscvproc.scala 352:{29,29}]
    mem_rd <= mux(reset, UInt<1>("h0"), _GEN_11) @[src/main/scala/riscvproc.scala 353:{29,29}]
    mem_wr <= mux(reset, UInt<1>("h0"), _GEN_12) @[src/main/scala/riscvproc.scala 354:{29,29}]
    mr_reg <= mux(reset, UInt<1>("h0"), _GEN_13) @[src/main/scala/riscvproc.scala 355:{29,29}]
    reg_file.clock <= clock
    reg_file.reset <= reset
    reg_file.io_reg_addr1 <= _reg_file_io_reg_addr1_T @[src/main/scala/riscvproc.scala 359:31]
    reg_file.io_reg_addr2 <= _reg_file_io_reg_addr2_T @[src/main/scala/riscvproc.scala 360:31]
    reg_file.io_wr_addr <= waddr @[src/main/scala/riscvproc.scala 361:29]
    reg_file.io_wr_data <= _reg_file_io_wr_data_T @[src/main/scala/riscvproc.scala 481:29]
    reg_file.io_wr_en <= reg_wr @[src/main/scala/riscvproc.scala 362:27]
    imm_gen.clock <= clock
    imm_gen.reset <= reset
    imm_gen.io_instr <= if_reg @[src/main/scala/riscvproc.scala 366:26]
    alu_ctrl.clock <= clock
    alu_ctrl.reset <= reset
    alu_ctrl.io_alu_op <= ctrl.io_alu_op @[src/main/scala/riscvproc.scala 371:28]
    alu_ctrl.io_funct73 <= _alu_ctrl_io_funct73_T_2 @[src/main/scala/riscvproc.scala 372:29]
    forwarding.clock <= clock
    forwarding.reset <= reset
    forwarding.io_id_ex_rs1 <= _forwarding_io_id_ex_rs1_T @[src/main/scala/riscvproc.scala 383:33]
    forwarding.io_id_ex_rs2 <= _forwarding_io_id_ex_rs2_T @[src/main/scala/riscvproc.scala 384:33]
    forwarding.io_id_ex_mem_read <= ctrl.io_memread @[src/main/scala/riscvproc.scala 385:38]
    forwarding.io_ex_mem_rd <= waddr @[src/main/scala/riscvproc.scala 388:33]
    forwarding.io_ex_mem_reg_write <= reg_wr @[src/main/scala/riscvproc.scala 389:40]
    forwarding.io_ex_mem_alu_result <= alu_reg @[src/main/scala/riscvproc.scala 387:41]
    forwarding.io_mem_wb_rd <= UInt<5>("h0") @[src/main/scala/riscvproc.scala 391:33]
    forwarding.io_mem_wb_reg_write <= UInt<1>("h0") @[src/main/scala/riscvproc.scala 392:40]
    forwarding.io_mem_wb_data <= reg_file.io_wr_data @[src/main/scala/riscvproc.scala 393:35]
    alu.clock <= clock
    alu.reset <= reset
    alu.io_ALUoperation <= alu_ctrl.io_operation @[src/main/scala/riscvproc.scala 421:29]
    alu.io_operand1 <= forward_alu_a @[src/main/scala/riscvproc.scala 422:25]
    alu.io_operand2 <= imm_out @[src/main/scala/riscvproc.scala 424:25]
    alu_reg <= mux(reset, UInt<32>("h0"), _GEN_1) @[src/main/scala/riscvproc.scala 379:{30,30}]
    wrd_mem <= mux(reset, UInt<32>("h0"), _GEN_2) @[src/main/scala/riscvproc.scala 380:{30,30}]
    data_mem.clock <= clock
    data_mem.reset <= reset
    data_mem.io_wr_en <= mem_wr @[src/main/scala/riscvproc.scala 477:27]
    data_mem.io_rd_en <= mem_rd @[src/main/scala/riscvproc.scala 478:27]
    data_mem.io_addr <= alu_reg @[src/main/scala/riscvproc.scala 475:26]
    data_mem.io_wr_data <= wrd_mem @[src/main/scala/riscvproc.scala 476:29]
