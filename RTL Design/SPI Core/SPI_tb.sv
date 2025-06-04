module SPI_tb;
    output [7:0] OutputSPI;
    reg clk_100MHz;       
    reg reset; 
    reg wrt; 
    reg [7:0] data_in;
    reg [1:0] instruction;
    reg [S-1:0] slave_sel;
    reg polarity;
    reg phase;
    reg spi_clk;
    reg Intermediate;

    // Instantiating the SPI_Core module
    SPI_Core UUT(
        .clk(clk_100MHz),
        .rst(reset),
        .write(wrt),
        .data_reg(data_in),
        .instr(instruction),
        .spi_miso(Intermediate),
        .spi_mosi(Intermediate),
        .spi_sclk(spi_clk),
        .spi_done(),
        .ss_n(slave_sel),
        .spi_out(OutputSPI),
        .dvsr_in(1024),
        .spi_ss_n()
    ); 

    initial begin
        // Initialize inputs
        clk_100MHz = 1'b0;
        reset = 1'b1;
        wrt = 1'b0;
        instruction = 2'b00;
        slave_sel = 4'b1110; 
        phase = 1'b0;
        polarity = 1'b0; 
        data_in = 8'b10110110;
        
        // Reset sequence
        #10 reset = 1'b0;
        
        // Stimulus
        #20 wrt = 1'b1;
        wrt = 1'b0;
        instruction = 2'b01;
        #20 instruction = 2'b10;
        #20 instruction = 2'b11;
        #20 instruction = 2'b00;
    end

    // Clock generation
    always #5 clk_100MHz = ~clk_100MHz;  

endmodule
