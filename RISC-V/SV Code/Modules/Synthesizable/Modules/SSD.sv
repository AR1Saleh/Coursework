module SSD (
    input                   clk_i, 
    input           [3:0]   x1, x2, x3, m0,

    output logic    [6:0]   Dspout, 
    output logic    [3:0]   Segout
);

reg [6:0] Registers [0:3];

Comb x1_(
    .SD1(Registers[0]),
    .num(x1)
); 

Comb x2_(
    .SD1(Registers[1]),
    .num(x2)
);

Comb x3_(
    .SD1(Registers[2]),
    .num(x3)
);

Comb m0_(
    .SD1(Registers[3]),
    .num(m0)
);

Mplex Out(
    .DspInt(Dspout),
    .SegInt(Segout),
    .clk(clk_i),
    .memDspInt(Registers)
);

endmodule

module Comb(
    output [6:0] SD1,    
    input [3:0] num 
);

assign SD1[0] = (((~num[3])&(~num[2])&(~num[1])&(num[0]))|((~num[3])&(num[2])&(~num[1])&(~num[0]))|((num[3])&(num[2])&(~num[1])&(num[0]))|(((num[3])&(~num[2])&(num[1])&(num[0]))));
assign SD1[1] = (((~num[3])&(num[2])&(~num[1])&(num[0]))|((num[3])&(num[2])&(~num[0]))|((num[2])&(num[1])&(~num[0]))|(((num[3])&(num[1])&(num[0]))));
assign SD1[2] = (((~num[3])&(~num[2])&(num[1])&(~num[0]))|((num[3])&(num[2])&(~num[0])|((num[3])&(num[2])&(num[1]))));
assign SD1[3] = (((~num[3])&(~num[2])&(~num[1])&(num[0]))|((~num[3])&(num[2])&(~num[1])&(~num[0]))|((num[3])&(~num[2])&(num[1])&(~num[0]))|((num[2])&(num[1])&(num[0])));           
assign SD1[4] = (((~num[3])&(num[0]))|((~num[3])&(num[2])&(~num[1]))|((~num[2])&(~num[1])&(num[0]))); 
assign SD1[5] = (((~num[3])&(~num[2])&(num[0]))|((~num[3])&(~num[2])&(num[1]))|((~num[3])&(num[1])&(num[0]))|((num[3])&(num[2])&(~num[1])&(num[0])));          
assign SD1[6] = (((~num[3])&(~num[2])&(~num[1]))|((num[3])&(num[2])&(~num[1])&(~num[0]))|((~num[3])&(num[2])&(num[1])&(num[0])));

endmodule


module Mplex(
    output logic [6:0] DspInt,
    output logic [3:0] SegInt,
    input logic  clk,
    input logic [6:0] memDspInt[0:3]
);

reg [1:0] count;

initial begin
count <= 0;
end

always @ (posedge clk) begin
    if (count == 0) begin
       SegInt <= 4'b1111;
    end
    SegInt[count - 1] <= 1'b1;
    DspInt <= memDspInt[count]; 
    SegInt[count] <= 1'b0;  
    count = count + 1;
end

endmodule
