module sram_two_port#(
    parameter DATA_BIT  = 16,
    parameter ADDR_BIT  = 8	,
    parameter ADDR_NUM  = 256
)(
    i0,
    addr,
    clk,
    wr,
    o0
);

input  [DATA_BIT-1:0] i0;
input  [ADDR_BIT-1:0] addr;
input                 clk;
input                 wr;
output [DATA_BIT-1:0] o0;

reg [DATA_BIT-1:0] mem [0:ADDR_NUM-1];
reg [DATA_BIT-1:0] mid_data ;

always@(posedge clk)
begin
    if(wr) mem[addr] <= i0; 
end

always@(posedge clk)
begin
    if(!wr) mid_data <= mem[addr]; 
end

assign o0 = mid_data; 

endmodule
