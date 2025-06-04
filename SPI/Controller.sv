module SPI (
    input logic [7:0]Din,
    input logic [15:0]ss_s_cycle,
    input logic [15:0]ss_h_cycle,
    input logic [15:0]ss_t_cycle,  
    input logic [15:0]dvsr,
    input logic clk, rst,
    input logic start, cpha, cpol, miso,
    output logic [7:0]Dout,
    output logic sclk,
    output logic spi_done_tick, ready,
    output logic mosi,
    output logic ss_n_out
);

typedef enum { Idle , ss_setup , cpha_delay , ss_hold , ss_turn , p0 , p1 } state;
state stt_reg, stt_next;
logic [15:0] c_reg, c_next;
logic [7:0] si_reg, si_next;
logic [7:0] so_reg, so_next;
logic [2:0] n_reg, n_next;//I count based on two modes, if phase is 0, then I increase at the middle of bit , else I count at the Loading of bit.
logic ready_i, spi_done_tick_i;
logic sclk_reg, sclk_next;
logic pclk;

always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        stt_reg <= Idle;
        c_reg <= 0;
        n_reg <= 0;
        si_reg <= 0;
        so_reg <= 0;
        sclk_reg <= 1'b0;
    end
    else begin
        stt_reg <= stt_next;
        c_reg <= c_next;
        n_reg <= n_next;
        si_reg <= si_next;
        so_reg <= so_next;
        sclk_reg <= sclk_next;
    end
end

always_comb begin 
    spi_done_tick_i = 1'b0;
    ready_i = 1'b0;
    stt_next = stt_reg;
    c_next = c_reg;
    n_next = n_reg;
    si_next = si_reg;
    so_next = so_reg;
    sclk_next = sclk_reg;
    case (stt_reg)
        
        Idle: begin
            ready_i = 1'b1;
            if (start) begin
                c_next = 0;
                n_next = 0;
                stt_next = ss_setup;    
            end
        end

        ss_setup: begin
            if (c_reg == ss_s_cycle) begin
                so_next = Din;
                ss_n_out = 1'b0;
                c_next = 0;
                stt_next = p0;
            end
            else c_next = c_next + 1;
        end            

        p0: begin //s_clk 0-to-1
            if (c_reg == dvsr) begin
                si_next = {miso, si_next[7:1]}; 
                c_next = 0;
                stt_next = p1;
            end
            else begin
                c_next = c_next + 1; 
            end
        end
        
        p1: begin //s_clk 1-to-0
            if(c_reg == dvsr) begin
                if (n_reg == 8) begin
                    stt_next = ss_hold;
                end    
                else begin
                    so_next = {1'b0, so_next[7:1]};
                    c_next = 0;
                    n_next = n_next + 1;
                    stt_next = p0;
                end
            end
        end
        
        ss_hold: begin
            if (c_reg == ss_h_cycle) begin
                    spi_done_tick_i = 1'b1;
                    c_next = 0;
                    stt_next = ss_turn;
                end
            else c_next = c_next + 1;        
        end
       
        ss_turn: begin
            ss_n_out = 1'b1;
            if (c_reg == ss_t_cycle) stt_next = Idle;
            else c_next = c_next + 1;
        end         
    endcase
end

assign spi_done_tick = spi_done_tick_i;
assign ready = ready_i;

assign pclk = (stt_next == p1 && ~cpha) || (stt_next == p0 && cpha); //I assume here that cpol is 0 by default.
//So when state is p1, then at cpha = 0, we get 1, else 0, and the reverse for state = p0. The reason why next stt's are used is to avoid delays.
assign sclk_next = (cpol) ? ~pclk : pclk; //This inverts the clk based on polarity.
//The signal is modified on the go, so we used a buffer(sclk_reg.)
assign Dout = si_reg;
assign mosi = so_reg[0];
assign sclk = sclk_reg;  

endmodule