`timescale 1ns / 1ns

module uart_tb(
    output Tx
);
    reg clk_100MHz;       
    reg reset;             
    reg wr_uart;          
    reg [7:0] data_in;
     

uart_test UUT(
    .clk(clk_100MHz),
    .reset(reset),
    .wr_uart(wr_uart),
    .data_in(data_in),
    .tx(Tx)
);

initial begin
    clk_100MHz = 1'b0;
    reset = 1'b1;
    #10 reset = 1'b0;
    data_in = 8'b00000000;
end

always #5 clk_100MHz = ~clk_100MHz; 

initial begin
    #100 data_in = 8'b01010100;
    wr_uart = 1'b1;
    #100000 data_in = 8'b11011111;
    wr_uart = 1'b0;
end 
endmodule
