
module Mux_2x1(
    input logic MemtoReg,
    input logic [31:0] in1,in2,
    output logic [31:0] out
    );
    assign out = MemtoReg ? in2:in1;
endmodule

