module riscv_tb;
  int i,j;

  bit         clk, rst_n;

  logic [6:0] Dspout;
  logic [3:0] Segout;
  
  initial begin

    reset_seq;

    for (i = 0; i< 16; i++) begin

        for (j = 0; j < 100_000; j++) begin
          @(posedge clk);  
        end

        $display("x1: %d\n", dut.inst.inst_reg.regfile[1]);
        $display("x2: %d\n", dut.inst.inst_reg.regfile[2]);
        $display("x3: %d\n\n", dut.inst.inst_reg.regfile[3]);        
    end

    $writememh("output.txt" , dut.inst.inst_reg.regfile);

    $stop;
  end
  
  // Reset task: uses active-low reset behavior.
  task automatic reset_seq;
    begin
      rst_n = 0;  
      #20;
      rst_n = 1;   
    end
  endtask
  
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end
  
  top dut(.*);  

endmodule
