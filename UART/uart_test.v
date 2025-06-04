`timescale 1ns / 1ns

module uart_test(
    input clk,       
    input reset,             
    input wr_uart,          // to start transmission
    input [7:0] data_in,
    output tx              
    );
    
    
    // Complete UART Core
    uart_top UART_UNIT
        (
            .clk_100MHz(clk),
            .reset(reset),
            .tx_start(wr_uart),
            .data_in(data_in),
            .tx(tx)
        );
    
endmodule
