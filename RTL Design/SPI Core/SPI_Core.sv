//When the processor reads data, the first 8 bits are data bits. and the 9th one is ready status signal.
//When the processor writes data, the first S-1 bits are the slave no. , next 8 bits are data bits. then the next 18 bits : first 16 is clk_divisor, 17th is cpol, 18th is cpha.
module SPI_Core 
    #(parameter S = 4, 
                t_setup = 512,
                t_hold = 512, 
                t_turn = 1024) 
(
    input logic clk,
    input logic reset,

    input logic write,//Write instruction.
    input logic cpol_i,
    input logic cpha_i,
    input logic spi_miso,
    input logic [1:0] instr,//I use this to first write the slave no. and the data bits, then I write the control bits(dvsr, cpol, cpha)
    input logic [S-1:0] ss_n,
    input logic [7:0] data_reg,
    input logic [15:0] dvsr_in,

    output logic spi_done,
    output logic spi_sclk,//This is used to synchronize with other chips(slaves).
    output logic spi_mosi,
    output logic [7:0] out_reg,//This is fetched from the SPI after transmission.
    output logic [S-1:0] spi_ss_n//The slave Selector.(01, 10). 11 being the null value. (Slaves are active-low.)
);

logic wr_en, wr_ss, wr_spi, wr_ctrl;
logic [S-1:0] ss_n_reg;
logic [7:0] spi_out;
logic spi_ready, cpol, cpha;
logic [15:0] dvsr;
logic ss_en;

SPI SPI_Controller(
    .clk(clk), 
    .rst(reset),
    .Din(data_reg[7:0]),
    .dvsr(dvsr),
    .start(wr_spi),
    .cpol(cpol),
    .cpha(cpha),
    .ss_h_cycle(t_hold),
    .ss_t_cycle(t_turn),
    .ss_s_cycle(t_setup),
    .Dout(spi_out),
    .sclk(spi_sclk),
    .miso(spi_miso),
    .mosi(spi_mosi),
    .spi_done_tick(),
    .ready(spi_ready),
    .ss_n_out(ss_en)
);

always_ff @( posedge clk, posedge reset ) begin
    if (reset) begin
        cpol <= 1'b0;
        cpha <= 1'b0;
        dvsr <= 15'h200;//Hexadecimal equivalent of 512.
        ss_n_reg <= {S{1'b1}};//Repeats 1 "S" times.
    end
    else begin
        if (wr_ctrl)
            dvsr <= dvsr_in[15:0];
            cpol <= cpol_i;
            cpha <= cpha_i;
        if(wr_ss)
            ss_n_reg <= ss_n[S-1:0];    
    end
end

assign wr_en = write;//First the Write instruction.
assign wr_ss = wr_en && instr[1:0]==2'b01;//Writing the Slave no.(The first S bits are now gone.)
assign wr_spi = wr_en && instr[1:0]==2'b10;//Writing to SPI.(Then the data bits.)
assign wr_ctrl = wr_en && instr[1:0]==2'b11;//Writing to Control Register.(Then the remaining 18 bits.)
//Further partitioning of the Control register.
//Using buffer to write Slave no.
assign spi_ss_n = ss_n_reg | {S{ss_en}};//When ss_en goes to 0. then we just do bitwise OR and see which line/path is equal to 0.
//The read data register.
assign out_reg = spi_out;
assign spi_done = spi_ready;
endmodule