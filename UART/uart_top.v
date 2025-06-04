`timescale 1ns / 1ns

// For 9600 baud with 100MHz FPGA clock: 
// 9600 * 16 = 153,600
// 100 * 10^6 / 153,600 = ~651      (counter limit M)
// log2(651) = 10                   (counter bits N) 


module uart_top
    #(
        parameter   DBITS = 8,          // number of data bits in a word
                    SB_TICK = 16,       // number of stop bit / oversampling ticks
                    BR_LIMIT = 651,     // baud rate generator counter limit
                    BR_BITS = 10        // number of baud rate generator counter bits
    )
    (
        input clk_100MHz,               
        input reset,                    
        input tx_start,               
        input [DBITS-1:0] data_in,     // data from Tx
        output tx                      // serial data out
    );
    
    wire tick;                         // sample tick from baud rate generator

    baud_rate_generator 
        #(
            .M(BR_LIMIT), 
            .N(BR_BITS)
         ) 
        BAUD_RATE_GEN   
        (
            .clk_100MHz(clk_100MHz), 
            .reset(reset),
            .tick(tick)
         );
    
    uart_transmitter
        #(
            .DBITS(DBITS),
            .SB_TICK(SB_TICK)
         )
         UART_TX_UNIT
         (
            .clk_100MHz(clk_100MHz),
            .reset(reset),
            .tx_start(tx_start),
            .sample_tick(tick),
            .data_in(data_in),
            .tx(tx)
         );
    
  
endmodule