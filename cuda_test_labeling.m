%% NEW VERS of CCL WITH SHARED-MEM
%% PARS
% dir/filenames
BASE_DIR    = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing';
bin_mat_fil = fullfile(BASE_DIR,'data','ALL.txt');
WRITE_TXT   = @(matname,ntx,nty,tdx,tdy) sprintf('%s-nt%sx%s-td%sx%s.txt',matname,num2str(ntx),num2str(nty),num2str(tdx),num2str(tdy));
% cuda code compile/run:
exefil      = fullfile(BASE_DIR,'Release','soil-sealing-2');
cu_compile  = 0; % Do you want MatLab compile your source .cu file?
save_mats   = 0; % Do you want to store BIN & MAT files for specific runs?
print_me    = 0; % Do you want .cu code print LAB-MAT at the end of every kernel?
%% set dim:
% NOTE:
%   Here it is assumed that the image can be partitioned into tiles having
%   1-pixel-width extra border along the tile perimeter. Therefore the
%   tiledimX is the length of the tile in X geographical dimension
%   including the extra border. Hence in order to be straight in sizing the
%   image, I fistly fix the tile dimensions (and the number of tiles, to
%   set the total size of the overall image) and then the number of rows
%   and columns is derived accordingly.

% fixed [no more then 32x32]:
tiledimX    = 32;  % 512 - 32 - 30
tiledimY    = 32;  % 2   - 32 - 30

% variable:
ntilesX     = 30;
ntilesY     = 32;
threshold   = 0.7; % to set the density of the image!

% consequence:
NC          = (ntilesX-1)*(tiledimX-1) + tiledimX -2;
NR          = (ntilesY-1)*(tiledimY-1) + tiledimY -2;
%% arguments to run within Nsight
fprintf('%d %d %d %d %d\n',tiledimX,tiledimY,NC,NR,print_me);
%% compute size on GPU
% size of BIN-MAT:
SIZE__(1)   = NR*NC*1;
% size of LAB-MAT:
SIZE__(2)   = ntilesX*ntilesY*tiledimX*tiledimY*4;
% print:
% fprintf('Size on GPU-memory:\t%.1f MB\n', sum( SIZE__ )/10^6)
%% create binary image

A                   = rand(NR,NC);

A(A>=(1-threshold)) = 1;
A(A<(1-threshold))  = 0;
% A               = logical(A);

%% save binary image

% fid         = fopen(fullfile(BASE_DIR,'data',bin_mat_fil),'w');
% for irow = 1:NR
%     for icol = 1:NC
%         fprintf(fid, '%d ', A(irow,icol));
%     end
%     fprintf(fid,'\n');
% end
% fclose(fid);

% faster but file has larger size
% save(fullfile(BASE_DIR,'data',bin_mat_fil), 'A', '-ascii')

% faster and small file size:
dlmwrite( bin_mat_fil, A, 'delimiter',' ')

%% compute MatLab labeled image

% compute labeled image from within MatLab;
tic,
Lml     = transpose( bwlabel(A',8) );
T(1)    = toc;

%% save MatLab labeled image
% save(fullfile(BASE_DIR,'data',WRITE_TXT('LAB_MAT-ml-',ntilesX,ntilesY,tiledimX,tiledimY)), 'A', '-ascii')
if save_mats
    dlmwrite(fullfile(BASE_DIR,'data',WRITE_TXT('LAB_MAT-ml-',ntilesX,ntilesY,tiledimX,tiledimY)),A,'delimiter',' ')
end
%% compile CUDA-code
% see: http://www.mathworks.com/matlabcentral/answers/47064-running-mex-files-compiled-on-linux
if cu_compile

    targfil         = fullfile(BASE_DIR,          'src','connected_component_labeling.cu');
    objfil          = fullfile(BASE_DIR,'Release','src','connected_component_labeling.o');
    d_fil           = fullfile(BASE_DIR,'Release','src','connected_component_labeling.d');
    comp_pars       = cell(3,1);
    comp_vers       = '/usr/local/cuda/bin/nvcc';
    comp_pars{1}    = sprintf('%s -O3 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "%s" "%s"',comp_vers,d_fil,targfil);
    comp_pars{2}    = sprintf('%s --compile -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "%s" "%s"',comp_vers,objfil,targfil);
    comp_pars{3}    = sprintf('%s --cudart static -link -o  "%s"  %s',comp_vers,exefil,objfil);

    % write compiling file:
    fprintf('Compiling:\n\t%s\n\n',targfil )
    % eval( ['!',comp_vers,' ',comp_pars_1,' ',objfil,' ',targfil] )
    % eval( ['!',comp_vers,' ',comp_pars_2,' ',exefil,' ',objfil] )
    fid = fopen(fullfile(BASE_DIR,'matlab','compile-fil'),'w');
    fprintf(fid,'%s\n','#!/bin/bash');
    for ii = 1:length(comp_pars)
        fprintf( fid, '%s\n', comp_pars{ii} );
    end
    fprintf(fid,'%s\n',['rm ',objfil]);
    fprintf(fid,'%s\n',['rm ',d_fil]);
    fclose(fid);

    % compile:
%     setenv('LD_LIBRARY_PATH', '');
    unix( ['sh ',fullfile(BASE_DIR,'matlab','compile-fil')] );
end
% % OLD
% comp_pars_1 = '--compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -x cu -o';
% comp_pars_2 = '--cudart static -link -o';
%% execute CUDA-code & grab stout

% run
run_pars        = [' ',num2str(tiledimX),' ',num2str(tiledimY),' ',num2str(NC),' ',num2str(NR),' ',num2str(print_me)];
run_str         = [exefil, run_pars];
fprintf('Running:\n\t%s\n\n',run_str);
tic
[status,out]    = unix( run_str, '-echo' );
T(3)            = toc;
T(2)            = str2double( out(strfind(out,':')+1:strfind(out,'[')-1) )/1000;

%% import CUDA labeled image
fid             = fopen(fullfile(BASE_DIR,'data','CUDA-code.txt'),'r');
Lcuda           = fscanf(fid, '%d ', [NC,NR]);
fclose(fid);
Lcuda           = Lcuda';
%% store images
if save_mats
    while ~exist( fullfile(BASE_DIR,'data','CUDA-code.txt'), 'file' ), end
    copyfile(fullfile(BASE_DIR,'data',bin_mat_fil),fullfile(BASE_DIR,'data',WRITE_TXT('BIN_MAT',ntilesX,ntilesY,tiledimX,tiledimY)))
    movefile(fullfile(BASE_DIR,'data','CUDA-code.txt'),fullfile(BASE_DIR,'data',WRITE_TXT('LAB_MAT-cu-',ntilesX,ntilesY,tiledimX,tiledimY)))
end
%% compare

DIFFER = Lcuda - Lml;
idx=find(DIFFER);
UL = unique([Lml(idx),Lcuda(idx)],'rows');
UC = unique([Lcuda(idx),Lml(idx)],'rows');

fprintf( '%s\n', repmat('*',1,55) );
fprintf( '   The Connected Component Labeling (CCL) procedure\n' );
fprintf( '\t   # CUDA vs MatLab # \n' );
fprintf( '   Image[%d,%d], tiledim[%d,%d], ntiles[%d,%d] \n',NR,NC,tiledimX,tiledimY,ntilesX,ntilesY );
fprintf( '   Size on GPU memory:\t%.1f MB\n', sum( SIZE__ )/10^6)
fprintf( '   differs in <<%d>> cells [norm=%.2f]\n', length(find(DIFFER)),norm(DIFFER) );
fprintf( '     > nID: %d (%s) vs %d (%s)\n',length(unique(Lcuda(:)))-1,'cuda',max(Lml(:)),'matlab' );
fprintf( '     > %d MatLab labels have more CUDA labels\n', sum(diff(UL(:,1))) ~= size(UL,1)-1 );
fprintf( '     > %d CUDA labels have more MatLab labels\n', sum(bwlabel(diff(UC(:,1)),4)) ~= size(UC,1)-1 );
fprintf( '     > speed-up = %3.2f\t( %3.2f / %3.2f [ms] )\n', T(1)/T(2),T(1)*1000,T(2)*1000 );
fprintf( '%s\n', repmat('*',1,55) );

%% plot

% figure(1),subplot(131),gpcolor(DIFFER),title('(CUDA - MatLab) CCL differences')
% figure(1),subplot(132),gpcolor(Lml),title('MatLab')
% figure(1),subplot(133),gpcolor(Lcuda),title('CUDA')

%% find a problem

eFound = unique( UL( find(diff(UL(:,1))==0), 1 ) );

for ii = 1:length(eFound)
    fprintf('MatLab ID[ %4d ] ---> [',eFound(ii));
    List = UL(UL(:,1)==eFound(ii),:);
    for jj=1:sum(UL(:,1)==eFound(ii))
        fprintf('%7d, ',List(jj,2) );
    end
    fprintf(']\n');
end

%% check number of pixels are the same (without accounting for objects)
fprintf('\n');
Lml_bin = Lml;
Lml_bin(Lml >= 1) = 1;
fprintf('Number of unrecognized object pixels: \t%10s%10d\n','[MatLab]',sum( A(:)-Lml_bin(:) )  );

Lcuda_bin = Lcuda;
Lcuda_bin(Lcuda >= 1) = 1;
fprintf('Number of unrecognized object pixels: \t%10s%10d\n','[CUDA]',sum( A(:)-Lcuda_bin(:) )  );

%% speedup
fprintf('\n');
fprintf('%8s%10s: %7.1f [ms]\n', 'Speed','[MatLab]', T(1)*1000 );
fprintf('%8s%10s: %7.1f [ms]\n', 'Speed','[CUDA]', T(2)*1000 );
fprintf('%8s%10s: %3.5f [--]\n', 'Speedup','[ml/cu]',T(1) / T(2) );
