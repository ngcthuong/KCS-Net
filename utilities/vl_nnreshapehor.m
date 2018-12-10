function y = vl_nnreshapehor(x, dzdy)
global blkSize
% VL_NNRESHAPE Feature reshaping
%   Y = VL_NNRESHAPE(X, DIMS) reshapes the input data X to have
%   the dimensions specified by DIMS. X is a SINGLE array of
%   dimension H x W x D x N where (H,W) are the height and width of
%   the map stack, D is the image depth (number of feature channels)
%   and N the number of of images in the stack. DIMS is a 1 x 3 array
%   of integers describing the dimensions that Y will take (batch
%   size is preserved). In addition to positive integers, the
%   following can also be specified in the style of caffe:
%
%   Interpretation of DIMS elements:
%   -1 := work it out from other dims
%    0 := copy dimension from X
%
%   NOTE: At most one dimension can be worked out from the others.
%
%   DZDX = VL_NNRESHAPE(X, DIMS, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.


if nargin <= 1 || isempty(dzdy)  % Forward pass    
    %y = single(zeros(w* blkSize, h * blkSize, 1, b));
    y = permute(x, [1, 3, 2, 4]); 
%     y = gpuArray(y); 
else    
    % default block size is 32x32
    %y = gpuArray(zeros(w/blkSize, h/blkSize, blkSize * blkSize, b));
    y = permute(dzdy, [1, 3, 2, 4]);     
%     y = single(y);
end