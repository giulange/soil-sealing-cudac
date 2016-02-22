%% understand how to do "zero" padding
% Any binary image given in input to gCCL (connected component labeling
% code written by Giuliano), has a whatever NC & NR numbers.
% This means that I have to pad in order to let cuda code works fine!
% When an image is passed to CUDA, it has a given number of columns and
% rows. Suppose they are:
NC          = 1220;
NR          = 1080;
% In the main (which can be in C or in JAVA) I decide to use the maximum
% available number of threads, which for my TESLA C-2075 is:
tiledimX    = 32;
tiledimY    = 32;
% According to these four parameters, I need to consider the following
% number of tiles:
ntilesX     = ceil( (NC+2-1) / (tiledimX-1) );
ntilesY     = ceil( (NR+2-1) / (tiledimY-1) );
% which requires to allocate an array of size NC1*NR1:
NC1 		= ntilesX*(tiledimX-1) +1;
NR1 		= ntilesY*(tiledimY-1) +1;
