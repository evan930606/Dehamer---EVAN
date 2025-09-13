module fifo_filt#(
    parameter DATA_BIT      = 9,
    parameter SRAM_ADDR_BIT = 8,
    parameter SRAM_ADDR_NUM = 256
)(
    clk,
    rst_n,
    i_rdy,
    i_ack,
    i_last,
    i_end,
    i0,
    o_rdy,
    o_ack,
    o_last,
    o_end,
    o0
);

localparam SRAM_DATA_BIT = 2 * DATA_BIT;

input                 clk;
input                 rst_n;
input                 i_rdy;
input                 i_ack;
input                 i_last;
input                 i_end;
input  [DATA_BIT-1:0] i0;
output                o_rdy;
output                o_ack;
output                o_last;
output                o_end;
output [DATA_BIT-1:0] o0;

wire                     sram_i_rdy;
wire                     sram_i_ack;
wire                     sram_i_last;
wire                     sram_i_end;
wire                     sram_o_rdy;
wire                     sram_o_ack;
wire                     sram_o_last;
wire                     sram_o_end;
wire                     sram_wr;
wire [SRAM_ADDR_BIT-1:0] sram_addr;
wire [SRAM_DATA_BIT-1:0] sram_i0;
wire [SRAM_DATA_BIT-1:0] sram_o0;

//---------------------------------------------------------------------------------
wire                  input_trig       = i_rdy & sram_o_ack;
reg                   input_state;
wire                  input_state_next = input_trig ? ~input_state : input_state;
reg  [DATA_BIT-1  :0] i0_tmp;
wire [DATA_BIT-1  :0] i0_tmp_next      = {i0};
wire [DATA_BIT*2-1:0] two_in_one_i0    = {i0, i0_tmp};
wire                  two_in_one_i_rdy = input_state & i_rdy;

assign                sram_i0          = two_in_one_i0;
assign                sram_i_rdy       = input_state;
assign                o_ack            = sram_o_ack | (!input_state);
assign                sram_i_last      = i_last;
assign                sram_i_end       = i_end;
//---------------------------------------------------------------------------------
reg  [1           :0] output_state;
reg  [1           :0] output_state_next;

always@(*)
begin
    case(output_state)
        2'b00 : output_state_next = sram_o_rdy         ? 2'b10 : 2'b00;
        2'b01 : output_state_next = sram_o_rdy & i_ack ? 2'b10 :
                                    o_end & i_ack      ? 2'b00 : 2'b01;
        2'b10 : output_state_next = i_ack              ? 2'b01 : 2'b10;
        default : output_state_next = 2'b00;
    endcase 
end

wire [DATA_BIT-1  :0] o0_tmp0    = sram_o0[DATA_BIT-1:0];
wire [DATA_BIT-1  :0] o0_tmp1    = sram_o0[DATA_BIT*2-1:DATA_BIT];
assign                o0         = (output_state == 2'b10) ? o0_tmp0 : o0_tmp1;
assign                o_rdy      = (output_state != 2'b00) & i_ack;
assign                sram_i_ack = (output_state == 2'b00) | (output_state == 2'b01) & i_ack;
assign                o_last     = sram_o_last;
assign                o_end      = sram_o_end;


fifo_filt_ctrl#(
    .ADDR_BIT(SRAM_ADDR_BIT)
)ctrl0(
    .clk      (clk        ),
    .rst_n    (rst_n      ),
    .i_rdy    (sram_i_rdy ),
    .i_ack    (sram_i_ack ),
    .i_last   (sram_i_last),
    .i_end    (sram_i_end ),
    .o_rdy    (sram_o_rdy ),
    .o_ack    (sram_o_ack ),
    .o_last   (sram_o_last),
    .o_end    (sram_o_end ),
    .sram_wr  (sram_wr    ),
    .sram_addr(sram_addr  )
);

sram_two_port#(
    .DATA_BIT(SRAM_DATA_BIT),
    .ADDR_BIT(SRAM_ADDR_BIT),
    .ADDR_NUM(SRAM_ADDR_NUM)
)sram0(
    .i0  (sram_i0  ),
    .addr(sram_addr),
    .clk (clk      ),
    .wr  (sram_wr  ),
    .o0  (sram_o0  ) 
);

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)            i0_tmp <= {DATA_BIT{1'b0}};
    else if(!input_state) i0_tmp <= i0_tmp_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)       input_state <= {1'b0};
    else             input_state <= input_state_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)       output_state <= {1'b0};
    else             output_state <= output_state_next;
end

endmodule