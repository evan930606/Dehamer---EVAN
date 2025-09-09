module Preprocessing#(
    parameter R_DATA_BIT = 8,
    parameter G_DATA_BIT = 8,
    parameter B_DATA_BIT = 8
)( 
    clk,
    rst_n,
    i_rdy,
    i_ack,
    i_last,
    i0_r,
    i0_g,
    i0_b,
    o_rdy,
    o_ack,
    o_last
    o0_r,
    o0_g,
    o0_b
);

localparam DATA_BIT = 9;

/* hankshake ---------------------------------------------------------------------------------------------------------------------------------
 |          | -> i_rdy  -> |             | -> o_rdy  -> |            |
 | upstream | <- o_ack  <- | this_module | <- i_ack  <- | downstream |
 |          | -> i_last -> |             | -> o_last -> |            |
--------------------------------------------------------------------------------------------------------------------------------------------*/
localparam PIPE_NUM = 1;
reg  [PIPE_NUM-1:0] pipeline_rdy;
wire [PIPE_NUM-1:0] pipeline_rdy_next  = {i_rdy, pipeline_rdy[PIPE_NUM-1:1]};
reg  [PIPE_NUM-1:0] pipeline_last;
wire [PIPE_NUM-1:0] pipeline_last_next = {i_last, pipeline_last[PIPEL_NUM-1:1]};
wire                pipeline_trig      = | pipeline_last ? i_ack : (i_rdy & i_ack);
assign              o_rdy              = pipeline_rdy[0];
assign              o_ack              = i_ack;
assign              o_last             = pipeline_last[0];
// 
localparam PIPE0_BIT = R_DATA_BIT + G_DATA_BIT + B_DATA_BIT;
reg  [PIPE0_BIT-1:0] pipeline0;
wire [PIPE0_BIT-1:0] pipeline0_next = {i0_r, i0_g, i0_b};
// 
wire [R_DATA_BIT-1:0] st1_i0_r = [PIPE0_BIT-1                       : PIPE0_BIT-R_DATA_BIT];
wire [G_DATA_BIT-1:0] st1_i0_g = [PIPE0_BIT-R_DATA_BIT-1            : PIPE0_BIT-R_DATA_BIT-G_DATA_BIT];
wire [B_DATA_BIT-1:0] st1_i0_b = [PIPE0_BIT-R_DATA_BIT-G_DATA_BIT-1 : PIPE0_BIT-R_DATA_BIT-G_DATA_BIT-B_DATA_BIT];

wire [DATA_BIT-1:0]   st1_i0_r_2d_9bit = {1'b0, st1_i0_r, {(DATA_BIT-1-R_DATA_BIT){1'b0}}};
wire [DATA_BIT-1:0]   st1_i0_g_2d_9bit = {1'b0, st1_i0_g, {(DATA_BIT-1-G_DATA_BIT){1'b0}}};
wire [DATA_BIT-1:0]   st1_i0_b_2d_9bit = {1'b0, st1_i0_b, {(DATA_BIT-1-B_DATA_BIT){1'b0}}};

assign                o0_r = st1_i0_r_2d_9bit + 2'b101100001;
assign                o0_g = st1_i0_g_2d_9bit + 2'b101101100;
assign                o0_b = st1_i0_b_2d_9bit + 2'b101110001;

//always
always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)              pipeline_rdy <= {PIPE_NUM{1'b0}};
    else if(pipeline_trig)  pipeline_rdy <= pipeline_rdy_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)              pipeline_last <= {PIPE_NUM{1'b0}};
    else if(pipeline_trig)  pipeline_last <= pipeline_last_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)              pipeline_last <= {PIPE_NUM{1'b0}};
    else if(pipeline_trig)  pipeline_last <= pipeline_last_next;
end

always@(posedge clk or negedge rst_n)
begin
    if(!rst_n)              pipeline0 <= {PIPE0_BIT{1'b0}};
    else if(pipeline_trig)  pipeline0 <= pipeline0_next;
end
endmodule