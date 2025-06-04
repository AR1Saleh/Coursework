//> using scala "2.13.12"
//> using dep "org.chipsalliance::chisel:6.7.0"
//> using plugin "org.chipsalliance:::chisel-plugin:6.7.0"
//> using options "-unchecked", "-deprecation", "-language:reflectiveCalls", "-feature", "-Xcheckinit", "-Xfatal-warnings", "-Ywarn-dead-code", "-Ywarn-unused", "-Ymacro-annotations"

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage

class RiscV extends Module {
  val io = IO(new Bundle {
    val x1 = Output(UInt(32.W))
    val x2 = Output(UInt(32.W))
    val x3 = Output(UInt(32.W))
  })

	////////////////////////////////////////////////////////////////
	// Submodule Definitions
	////////////////////////////////////////////////////////////////

	class InstructionMemory extends Module {
	val io = IO(new Bundle {
	val addr = Input(UInt(32.W))
	val instr = Output(UInt(32.W))
	})
	val mem = Mem(16, UInt(32.W))
	val defaultProgram = VecInit(Seq(
	"h00a00093".U, "h00102023".U, "h00002103".U, "h00100093".U,
	"h00500113".U, "h002081b3".U, "h00100093".U, "h00100113".U,
	"h00208663".U, "h00000013".U, "h00000013".U, "h00108093".U, 
	"h00100093".U, "h00102023".U, "h00002103".U, "h00110193".U, 
	))
	defaultProgram.zipWithIndex.foreach { case (instr, idx) =>
	mem.write(idx.U, instr)
	}
	io.instr := mem((io.addr >> 2)(3, 0))
	}

	class DataMemory extends Module {
	val io = IO(new Bundle {
	val wr_en = Input(Bool())
	val rd_en = Input(Bool())
	val addr = Input(UInt(32.W))
	val wr_data = Input(UInt(32.W))
	val rdata = Output(UInt(32.W))
	})

	// Fix: Added missing closing parentheses
	val mem = RegInit(VecInit(Seq.fill(2)(0.U(32.W)))) // <- 3 closing )

	when(!reset.asBool) {
	when(io.wr_en) {
	mem(io.addr(0)) := io.wr_data
	}
	}
	io.rdata := Mux(io.rd_en, mem(io.addr(0)), 0.U)
	}

	class Registers extends Module {
	val io = IO(new Bundle {
	val reg_addr1 = Input(UInt(5.W))
	val reg_addr2 = Input(UInt(5.W))
	val wr_addr = Input(UInt(5.W))
	val wr_data = Input(UInt(32.W))
	val wr_en = Input(Bool())
	val reg_data1 = Output(UInt(32.W))
	val reg_data2 = Output(UInt(32.W))

	// New debug ports
	val x1 = Output(UInt(32.W))
	val x2 = Output(UInt(32.W))
	val x3 = Output(UInt(32.W))
	})

	// Fix: Added missing closing parentheses
	val regfile = RegInit(VecInit(Seq.fill(32)(0.U(32.W)))) // <- 3 closing )

	when(io.wr_en && io.wr_addr =/= 0.U) {
	regfile(io.wr_addr) := io.wr_data
	}
	io.reg_data1 := Mux(io.reg_addr1 === 0.U, 0.U, regfile(io.reg_addr1))
	io.reg_data2 := Mux(io.reg_addr2 === 0.U, 0.U, regfile(io.reg_addr2))

	// Connect debug ports
	io.x1 := regfile(1)
	io.x2 := regfile(2)
	io.x3 := regfile(3)
	}

	class ImmGen extends Module {
	val io = IO(new Bundle {
	val instr = Input(UInt(32.W))
	val imm_val = Output(UInt(32.W))
	})

	val op_code = io.instr(6, 0)

	io.imm_val := 0.U  // Default value

	switch(op_code) {
	// I-type (load/immediate operations)
	is("b0000011".U, "b0010011".U) {
	io.imm_val := Cat(Fill(20, io.instr(31)), io.instr(31, 20))
	}

	// S-type (store)
	is("b0100011".U) {
	io.imm_val := Cat(Fill(20, io.instr(31)), io.instr(31, 25), io.instr(11, 7))
	}

	// B-type (branch)
	is("b1100011".U) {
	io.imm_val := Cat(
	    Fill(20, io.instr(31)),
	    io.instr(7),
	    io.instr(30, 25),
	    io.instr(11, 8),
	    0.U(1.W)
	)
	}

	// U-type (LUI/AUIPC)
	is("b0110111".U, "b0010111".U) {
	io.imm_val := Cat(io.instr(31, 12), 0.U(12.W))
	}

	// J-type (JAL)
	is("b1101111".U) {
	io.imm_val := Cat(
	    Fill(11, io.instr(31)),
	    io.instr(31),
	    io.instr(19, 12),
	    io.instr(20),
	    io.instr(30, 21),
	    0.U(1.W)
	)
	}

	// JALR
	is("b1100111".U) {
	io.imm_val := Cat(Fill(20, io.instr(31)), io.instr(31, 20))
	}
	}
	}

	class Controller extends Module {
	val io = IO(new Bundle {
	val opcode   = Input(UInt(7.W))
	val branch   = Output(Bool())
	val alu_src  = Output(Bool())
	val regwrite = Output(Bool())
	val memread  = Output(Bool())
	val memwrite = Output(Bool())
	val memtoreg = Output(Bool())
	val alu_op   = Output(UInt(2.W))
	})

	// Default control values
	io.alu_src  := false.B
	io.memtoreg := false.B
	io.regwrite := false.B
	io.memread  := false.B
	io.memwrite := false.B
	io.branch   := false.B
	io.alu_op   := 0.U(2.W)

	switch(io.opcode) {
	// Loads (I-type)
	is("b0000011".U) {
	io.alu_src  := true.B
	io.memtoreg := true.B
	io.regwrite := true.B
	io.memread  := true.B
	}

	// Immediate operations (I-type)
	is("b0010011".U) {
	io.alu_src  := true.B
	io.regwrite := true.B
	}

	// Stores (S-type)
	is("b0100011".U) {
	io.alu_src  := true.B
	io.memwrite := true.B
	}

	// R-type operations
	is("b0110011".U) {
	io.regwrite := true.B
	io.alu_op   := "b10".U(2.W)
	}

	// Branches (B-type)
	is("b1100011".U) {
	io.branch := true.B
	io.alu_op := "b01".U(2.W)
	}
	}
	}


	class ForwardingUnit extends Module {
	  val io = IO(new Bundle {
	    // From ID/EX Pipeline Register
	    val id_ex_rs1 = Input(UInt(5.W))
	    val id_ex_rs2 = Input(UInt(5.W))
	    val id_ex_mem_read = Input(Bool())  // Is load instruction?
	    
	    // From EX/MEM Pipeline Register
	    val ex_mem_rd = Input(UInt(5.W))
	    val ex_mem_reg_write = Input(Bool())
	    val ex_mem_alu_result = Input(UInt(32.W))
	    
	    // From MEM/WB Pipeline Register
	    val mem_wb_rd = Input(UInt(5.W))
	    val mem_wb_reg_write = Input(Bool())
	    val mem_wb_data = Input(UInt(32.W))
	    
	    // Forwarding Controls
	    val forward_a = Output(UInt(2.W))
	    val forward_b = Output(UInt(2.W))
	    
	    val load_use_hazard = Output(Bool())
	  })

	  // Forwarding logic for RS1
	  io.forward_a := 0.U
	  when(io.ex_mem_reg_write && (io.ex_mem_rd === io.id_ex_rs1) && (io.ex_mem_rd =/= 0.U)) {
	    io.forward_a := 2.U  // Forward EX/MEM ALU result
	  }.elsewhen(io.mem_wb_reg_write && (io.mem_wb_rd === io.id_ex_rs1) && (io.mem_wb_rd =/= 0.U)) {
	    io.forward_a := 1.U  // Forward MEM/WB data
	  }

	  // Forwarding logic for RS2
	  io.forward_b := 0.U
	  when(io.ex_mem_reg_write && (io.ex_mem_rd === io.id_ex_rs2) && (io.ex_mem_rd =/= 0.U)) {
	    io.forward_b := 2.U  // Forward EX/MEM ALU result
	  }.elsewhen(io.mem_wb_reg_write && (io.mem_wb_rd === io.id_ex_rs2) && (io.mem_wb_rd =/= 0.U)) {
	    io.forward_b := 1.U  // Forward MEM/WB data
	  }

	  // Marker for load-use hazard
	  io.load_use_hazard := io.id_ex_mem_read && 
		       ((io.id_ex_rs1 === io.ex_mem_rd) || 
		       (io.id_ex_rs2 === io.ex_mem_rd)) &&
		       (io.ex_mem_rd =/= 0.U)
	}	

	class ALU extends Module {
	val io = IO(new Bundle {
	val ALUoperation = Input(UInt(4.W))
	val operand1 = Input(UInt(32.W))
	val operand2 = Input(UInt(32.W))
	val ALUresult = Output(UInt(32.W))
	val zero = Output(Bool())
	})

	// ALU operation selection
	io.ALUresult := 0.U
	switch(io.ALUoperation) {
	is("b0000".U) { io.ALUresult := io.operand1 & io.operand2 }
	is("b0001".U) { io.ALUresult := io.operand1 | io.operand2 }
	is("b0010".U) { io.ALUresult := io.operand1 + io.operand2 }
	is("b0110".U) { io.ALUresult := io.operand1 - io.operand2 }
	}

	// Zero flag generation
	io.zero := (io.ALUoperation === "b0110".U(4.W)) && 
		(io.ALUresult === 0.U(32.W))
	}

	class ALUCtrl extends Module {
	val io = IO(new Bundle {
	val alu_op = Input(UInt(2.W))
	val funct73 = Input(UInt(4.W))
	val operation = Output(UInt(4.W))
	})

	// Default operation (ADD)
	io.operation := "b0010".U(4.W)

	switch(io.alu_op) {
	is("b00".U) {  // Immediate/load/store (ADD)
	io.operation := "b0010".U(4.W)
	}
	is("b01".U) {  // Branch comparison (SUB)
	io.operation := "b0110".U(4.W)
	}
	is("b10".U) {  // R-type instructions
	switch(io.funct73) {
	    is("b0000".U(4.W)) {  // ADD
	    io.operation := "b0010".U(4.W)
	    }
	    is("b1000".U(4.W)) {  // SUB
	    io.operation := "b0110".U(4.W)
	    }
	    is("b0111".U(4.W)) {  // AND
	    io.operation := "b0000".U(4.W)
	    }
	    is("b0110".U(4.W)) {  // OR
	    io.operation := "b0001".U(4.W)
	    }
	}
	}
	}
	}

	// Convert reset to active-low
	val rst_n = !reset.asBool

	////////////////////////////////////////////////////////////////
	// Submodule Declarations
	////////////////////////////////////////////////////////////////
	val pc_current = RegInit(0.U(32.W))
	val pc_next = Wire(UInt(32.W))

	// Program Counter Logic
	pc_current := pc_next

	val f_pc = RegInit(0.U(32.W))
	when(rst_n) {
	f_pc := pc_current
	}.otherwise {
	f_pc := 0.U
	}

	// PC Adder (PC + 4)
	val result = pc_current + 4.U

	// Instruction Memory
	val instr_mem = Module(new InstructionMemory)
	instr_mem.io.addr := pc_current
	val instr = instr_mem.io.instr

	// IF/ID Pipeline Register
	val if_reg = RegInit(0.U(32.W))

	// Control Unit
	val ctrl = Module(new Controller)
	ctrl.io.opcode := if_reg(6,0)
	val branch = ctrl.io.branch
	val alu_src = ctrl.io.alu_src
	val regwrite = ctrl.io.regwrite
	val memread = ctrl.io.memread
	val memwrite = ctrl.io.memwrite
	val memtoreg = ctrl.io.memtoreg
	val alu_op = ctrl.io.alu_op

	// ID/EX Pipeline Registers
	val waddr = RegInit(0.U(5.W))
	val reg_wr = RegInit(false.B)
	val mem_rd = RegInit(false.B)
	val mem_wr = RegInit(false.B)
	val mr_reg = RegInit(false.B)

	// Register File
	val reg_file = Module(new Registers)
	reg_file.io.reg_addr1 := if_reg(19,15)
	reg_file.io.reg_addr2 := if_reg(24,20)
	reg_file.io.wr_addr := waddr
	reg_file.io.wr_en := reg_wr

	// Immediate Generator
	val imm_gen = Module(new ImmGen)
	imm_gen.io.instr := if_reg
	val imm_val = imm_gen.io.imm_val

	// ALU Control
	val alu_ctrl = Module(new ALUCtrl)
	alu_ctrl.io.alu_op := alu_op
	alu_ctrl.io.funct73 := Cat(if_reg(30), if_reg(14,12))

	// Instantiate Forwarding Unit
	val forwarding = Module(new ForwardingUnit)
	val alu = Module(new ALU)
	
	// EX/MEM Pipeline Registers
	val alu_reg = RegInit(0.U(32.W))
	val wrd_mem = RegInit(0.U(32.W))
	
	// Connect forwarding unit inputs
	forwarding.io.id_ex_rs1 := if_reg(19,15)
	forwarding.io.id_ex_rs2 := if_reg(24,20)
	forwarding.io.id_ex_mem_read := memread
	
	forwarding.io.ex_mem_alu_result := alu_reg
	forwarding.io.ex_mem_rd := waddr
	forwarding.io.ex_mem_reg_write := reg_wr
	
	forwarding.io.mem_wb_rd := 0.U  // Not used in 3-stage
	forwarding.io.mem_wb_reg_write := false.B  // Not used in 3-stage
	forwarding.io.mem_wb_data := reg_file.io.wr_data
	
	// Forwarding Muxes
	val forward_alu_a = MuxCase(
	  reg_file.io.reg_data1,  // Default value
	  Seq(
	    (forwarding.io.forward_a === 1.U) -> alu_reg,       // Forward EX/MEM result
	    (forwarding.io.forward_a === 2.U) -> reg_file.io.wr_data  // Forward WB result
	  )
	)

	val forward_alu_b = MuxCase(
	  reg_file.io.reg_data2,  // Default value 
	  Seq(
	    (forwarding.io.forward_b === 1.U) -> alu_reg,       // Forward EX/MEM result
	    (forwarding.io.forward_b === 2.U) -> reg_file.io.wr_data  // Forward WB result
	  )
	)
	
	when(rst_n) {
	alu_reg := alu.io.ALUresult
	wrd_mem := forward_alu_b
	}.otherwise {
	alu_reg := 0.U
	wrd_mem := 0.U
	}
	
	// Modified ALU connections
	alu.io.ALUoperation := alu_ctrl.io.operation
	alu.io.operand1 := forward_alu_a
	val imm_out = Mux(alu_src, imm_val, forward_alu_b)
	alu.io.operand2 := imm_out

	// Branch Offset Calculation
	val addresult = f_pc + imm_val

	// Modified Stall Logic
	val stall = forwarding.io.load_use_hazard

	// WRONG: You were injecting NOP as PC value
	// pc_next := Mux(stall, 0x00000013, ...)
	// RIGHT: Freeze PC but inject NOP in pipeline
	pc_next := Mux(
		stall, 
		pc_current, 
		Mux(branch && alu.io.zero, addresult, result)
	)
	
	// ID/EX pipeline register section
	when(rst_n) {
	    // Bubble Injection in ID/EX Registers
	    when(stall) {
		// Kill ID/EX pipeline register
		waddr := 0.U
		reg_wr := false.B
		mem_rd := false.B
		mem_wr := false.B
		mr_reg := false.B
		
		// Inject NOP into IF/ID
		if_reg := 0x00000013.U  // Actual NOP instruction
	    }.otherwise {
		waddr := if_reg(11,7)
		reg_wr := regwrite
		mem_rd := memread
		mem_wr := memwrite
		mr_reg := memtoreg
		
		if_reg := instr
	    }
	}.otherwise {
	    waddr := 0.U
	    reg_wr := false.B
	    mem_rd := false.B
	    mem_wr := false.B
	    mr_reg := false.B
		
	    if_reg := 0.U		
	}

	// Data Memory
	val data_mem = Module(new DataMemory)
	data_mem.io.addr := alu_reg
	data_mem.io.wr_data := wrd_mem
	data_mem.io.wr_en := mem_wr
	data_mem.io.rd_en := mem_rd

	// Writeback Mux
	reg_file.io.wr_data := Mux(mr_reg, data_mem.io.rdata, alu_reg)

	// Connect debug signals
	io.x1 := reg_file.io.x1
	io.x2 := reg_file.io.x2
	io.x3 := reg_file.io.x3

}

object Main extends App {
  println(
    ChiselStage.emitSystemVerilog(
      gen = new RiscV,
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}
