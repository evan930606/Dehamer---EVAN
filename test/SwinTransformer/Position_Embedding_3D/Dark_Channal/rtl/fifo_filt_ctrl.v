module fifo_filt_ctrl#(
    parameter ADDR_BIT = 8
)(
    clk,
    rst_n,
    i_rdy,
    i_ack,
    i_last,
    i_end,
    o_rdy,
    o_ack,
    o_last,
    o_end,
    sram_wr,
    sram_addr
);

input                 clk;
input                 rst_n;
input                 i_rdy;
input                 i_ack;
input                 i_last;
input                 i_end;
output                o_rdy;
output                o_ack;
output                o_last;
output                o_end;
output                sram_wr;
output [ADDR_BIT-1:0] sram_addr;

// state ------------------------------------------------------------------------------------------------------------------------
reg    [1           :0] state;
reg    [1           :0] state_next;

reg    [ADDR_BIT    :0] cnt;
reg    [ADDR_BIT    :0] full_num;
wire   [ADDR_BIT    :0] full_num_next = cnt;
wire                    full     = (cnt == full_num) | (state == 2'b00) & i_last? 1'b1 : 1'b0;
wire                    empty    = (cnt == {ADDR_BIT{1'b0}}) ? 1'b1 : 1'b0;
wire                    write    = (state == 2'b10) ? 1'b0 : i_rdy & (!full);
wire                    read     = (state == 2'b00) ? 1'b0:
                                   (state == 2'b01) ? full & i_ack : i_ack;

wire   [ADDR_BIT  :0]   cnt_inc  = write ? cnt + 1'b1 : 
                                   read  ? cnt - 1'b1 : cnt;
wire   [ADDR_BIT  :0]   cnt_next = cnt_inc; 

reg    [ADDR_BIT-1:0]   sram_write_addr;
wire   [ADDR_BIT  :0]   sram_write_addr_inc  = write ? sram_write_addr + 1'b1 : sram_write_addr;
wire                    sram_write_addr_end  = sram_write_addr_inc == full_num ? 1'b1 : 1'b0;
wire   [ADDR_BIT-1:0]   sram_write_addr_next = sram_write_addr_end ? {ADDR_BIT{1'b0}} : sram_write_addr_inc[ADDR_BIT-1:0];

reg    [ADDR_BIT-1:0]   sram_read_addr;
wire   [ADDR_BIT  :0]   sram_read_addr_inc   = read ? sram_read_addr + 1'b1 : sram_read_addr;
wire                    sram_read_addr_end   = sram_read_addr_inc == full_num ? 1'b1 : 1'b0;
wire   [ADDR_BIT-1:0]   sram_read_addr_next  = sram_read_addr_end ? {ADDR_BIT{1'b0}} : sram_read_addr_inc[ADDR_BIT-1:0];


always@(*)
begin
    case(state)
        2'b00 : state_next = i_last ? 2'b01 : 2'b00;
        2'b01 : state_next = i_end  ? 2'b10 : 2'b01;
        2'b10 : state_next = empty  ? 2'b00 : 2'b10;
        default : state_next = 2'b00;
    endcase
end

assign                o_rdy     = read;
assign                o_ack     = (state == 2'b00) ? 1'b1:
                                  (state == 2'b01 & !full) ? 1'b1 : 1'b0;
assign                o_last    = (state == 2'b01) & i_last | (state == 2'b10) & empty;
assign                o_end     = (state == 2'b10) & empty;
assign                sram_wr   = write;
assign                sram_addr = write ? sram_write_addr : sram_read_addr;


always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)               full_num <= {1'b1, {ADDR_BIT{1'b0}}};
    else if(i_last)          full_num <= full_num_next;
    else if(state == 2'b0)   full_num <= {1'b1, {ADDR_BIT{1'b0}}};
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n) cnt <= {ADDR_BIT{1'b0}};
    else       cnt <= cnt_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n) state <= 2'b0;
    else       state <= state_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n) sram_write_addr <= {ADDR_BIT{1'b0}};
    else       sram_write_addr <= sram_write_addr_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n) sram_read_addr <= {ADDR_BIT{1'b0}};
    else       sram_read_addr <= sram_read_addr_next;
end

endmodule