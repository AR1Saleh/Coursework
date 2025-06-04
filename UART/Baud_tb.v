`timescale 1ns / 1ns

module Baud_tb(
    output tick
);

    reg clk_100MHz;       
    reg reset;                  

baud_rate_generator UUT(
    .clk_100MHz(clk_100MHz),
    .reset(reset),
    .tick(tick)
);

initial begin
    clk_100MHz = 1'b0;
    reset = 1'b1;
    #2000000 reset = 1'b0;
end

always #5 clk_100MHz = ~clk_100MHz; 

endmodule