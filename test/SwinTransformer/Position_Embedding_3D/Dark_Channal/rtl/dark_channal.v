module dark_channal(
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
    o_last,
    o0
);

localparam DATA_BIT = 9;

input                 clk;
input                 rst_n;
input                 i_rdy;
input                 i_ack;
input                 i_last;
input                 i_end;
input  [DATA_BIT-1:0] i0_r;
input  [DATA_BIT-1:0] i0_g;
input  [DATA_BIT-1:0] i0_b;
output                o_rdy;
output                o_ack;
output                o_last;
output                o_end;
input  [DATA_BIT-1:0] i0_r;
output [DATA_BIT-1:0] o0;



endmodule