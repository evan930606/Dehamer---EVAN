`timescale 1ns/100ps
module tb;

real     TESTBENCH_CYCLE = 10.0;
real     QUARTER         = 2.5;

reg                clk      = 1'b1;
reg                rst_n    = 1'b1;

always    #(TESTBENCH_CYCLE / 2.0)    clk = ~clk;

//
parameter R_DATA_BIT = 8;
parameter G_DATA_BIT = 8;
parameter B_DATA_BIT = 8;
parameter DATA_BIT   = 9;
parameter C_NUM  = 3;
parameter H_NUM  = 480;
parameter W_NUM  = 640;

//--------DUT
integer   fin_r;
integer   fout_w;
reg                     i_rdy;
reg                     i_ack;
reg                     i_last;
reg    [R_DATA_BIT-1:0] i0_r;
reg    [G_DATA_BIT-1:0] i0_g;
reg    [B_DATA_BIT-1:0] i0_b;
wire                    o_rdy;
wire                    o_ack;
wire                    o_last;
wire   [DATA_BIT-1:0]   o0_r;
wire   [DATA_BIT-1:0]   o0_g;
wire   [DATA_BIT-1:0]   o0_b;

//--------ans
integer   fans_r;
reg       [DATA_BIT-1:0] ans_r;
reg       [DATA_BIT-1:0] ans_g;
reg       [DATA_BIT-1:0] ans_b;
reg       [31:0]         c;
reg       [31:0]         h;
reg       [31:0]         w;
reg       [31:0]         o_h;
reg       [31:0]         o_w;
//--------mem
integer   read;
reg       stop;
reg       while_trig;

/* input */
Preprocessing#(
    .R_DATA_BIT(R_DATA_BIT),
    .G_DATA_BIT(G_DATA_BIT),
    .B_DATA_BIT(B_DATA_BIT)
)top0( 
    .clk   (clk   ),
    .rst_n (rst_n ),
    .i_rdy (i_rdy ),
    .i_ack (i_ack ),
    .i_last(i_last),
    .i0_r  (i0_r  ),
    .i0_g  (i0_g  ),
    .i0_b  (i0_b  ),
    .o_rdy (o_rdy ),
    .o_ack (o_ack ),
    .o_last(o_last),
    .o0_r  (o0_r  ),
    .o0_g  (o0_g  ),
    .o0_b  (o0_b  )
);


task waveform;
begin
    $dumpfile("./wave.vcd");
    $dumpvars;
end
endtask

task global_reset;
begin
    #(6*TESTBENCH_CYCLE);
    rst_n = 1'b0;
    #(30*TESTBENCH_CYCLE);
    rst_n = 1'b1;
    $display(" [Msg] Global Reset Done.");
end
endtask

task file;
begin
    fin_r  = $fopen({"../data/","input_binary_3_480_640.txt"},"r");
    fans_r = $fopen({"../data/","ans_binary_3_480_640.txt"},"r");
    fout_w = $fopen({"../data/","output_binary_3_480_640.txt"},"w");
end
endtask

task run;
begin
    repeat(1) @(negedge clk);
    h = 32'b0;
    w = 32'd0;
    o_h = 32'b0;
    o_w = 32'd0;
    i_rdy  = 1'b0;
    i_ack  = 1'b0;
    i_last = 1'b0;
    while_trig = 1'b0;
    repeat(1) @(posedge clk);
    while(while_trig == 1'b0) begin
        #QUARTER;
        repeat(1) @(negedge clk);
        // signal
        i_rdy  = 1'b0;
        i_ack  = 1'b0;
        if(h != H_NUM) begin
            i_rdy = $urandom_range(0, 1);
            i_ack = $urandom_range(0, 1);
        end

        #QUARTER;

        if(o_ack == 1'b1 && i_rdy == 1'b1) begin
            read = $fscanf(fin_r, "%b", i0_r);
            read = $fscanf(fin_r, "%b", i0_g);
            read = $fscanf(fin_r, "%b", i0_b);
        end

        if ((o_rdy == 1'b1) && (i_ack == 1'b1)) begin
            $fwrite(fout_w, "%b\n", o0_r);
            $fwrite(fout_w, "%b\n", o0_g);
            $fwrite(fout_w, "%b\n", o0_b);
            read = $fscanf(fans_r,"%b", ans_r);
            read = $fscanf(fans_r,"%b", ans_g);
            read = $fscanf(fans_r,"%b", ans_b);
            if(o_last == 1'b1) while_trig = 1'b1;

            if(o_w == W_NUM-1) begin 
                o_w = 32'b0;
                if(o_h != H_NUM-1) o_h = o_h + 1'b1;
            end
            else           o_w = o_w + 1'b1;
        end

        if(h == H_NUM-1 && w == W_NUM-1) i_last = 1'b1;
        else                             i_last = 1'b0;

        if(o_ack == 1'b1 && i_rdy == 1'b1) begin
            if(w == W_NUM-1) begin 
                w = 32'b0;
                if(h < H_NUM-1) h = h + 1'b1;
            end
            else           w = w + 1'b1;
        end

    end
    $fclose(fin_r);
    $fclose(fout_w);
    $fclose(fans_r);
end
endtask

initial begin
    waveform;
    global_reset;
    file;
    run;
    #3000 $finish;
end

endmodule