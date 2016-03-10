%% NEW VERS of CCL WITH SHARED-MEM
clear,clc
BASE_DIR        = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing';
T               = NaN(2,1);
% –/O
FIL_LAB         = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/LAB-MAT-cuda.tif';
FIL_LAB_ssgci   = fullfile(BASE_DIR,'data','ccl_1toN_hist_lab.tif');
FIL_LABrand     = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/Lcuda_random.tif';

FIL_HIST        = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/cu_histogram.txt';
FIL_IDra        = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/ID_rand_cpu.txt';
FIL_ID1N        = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/ID_1toN_cpu.txt';
WRITE_TXT       = @(matname,ntx,nty,tdx,tdy) sprintf('%s-nt%sx%s-td%sx%s.txt',matname,num2str(ntx),num2str(nty),num2str(tdx),num2str(tdy));
% cuda code compile/run:
exefil          = fullfile(BASE_DIR,'Release','soil-sealing-2');
%% GDAL conversion
% gdal_translate -ot Byte -co TILED=YES -co BLOCKXSIZE=1504 -co BLOCKYSIZE=1216 -co COMPRESS=PACKBITS imp_mosaic_char_2006_cropped2_roi.tif imp_mosaic_char_2006_cropped2_roi__.tif
%% EXPLANATION ON HOW TO USE PARS
% If you need to test the cuda kernel massively on different types of data
% input in terms of both size and density you have to select the following
% parameters:
% 
% **fixed**
% FIL_ROI         = fullfile(BASE_DIR,'data','created-on-the-fly_ROI.tif');
% FIL_BIN         = fullfile(BASE_DIR,'data','created-on-the-fly_BIN.tif');
% cu_compile      = 0; % Do you want MatLab compile your source .cu file?
% cu_run          = 1; % Do you want to run the cuda code (compiled here or outside)?
% save_mats       = 0; % Do you want to store BIN & MAT files for specific runs?
% print_me        = 0; % Do you want .cu code print LAB-MAT at the end of every kernel?
% create_bin      = 1; % Do you want to create a new BIN array for a new comparison?
% deep_check      = 0/1; % Do you want to perform a deep comparison?
% **to-be-changed**
% ntilesX         = 95;
% ntilesY         = 133;
% threshold       = 0.7; % set image sparsity: near 0 high density, near 1 high sparsity
% 
% In order to full check the histogram counts you need to use the option #2
% in the cell named "compare | histogram" and deactivate the option #1 !!
%% PARS
T(2)            = 0.0;% [s]
% I/–
%   create on-the-fly:
% FIL_ROI         = fullfile(BASE_DIR,'data','created-on-the-fly_ROI.tif');
% FIL_BIN         = fullfile(BASE_DIR,'data','created-on-the-fly_BIN.tif');
%   created on the ss-gci
FIL_ROI         = fullfile(BASE_DIR,'data','ccl_1toN_hist_roi.tif');
FIL_BIN         = fullfile(BASE_DIR,'data','ccl_1toN_hist_bin.tif');

%   other data:
% FIL_ROI       = fullfile('/home/giuliano/git/cuda/fragmentation/data','ROI.tif');
% FIL_BIN       = fullfile('/home/giuliano/git/cuda/fragmentation/data','BIN.tif');
% FIL_ROI       = '/home/giuliano/git/cuda/fragmentation/data/lodi1954_roi.tif';
% FIL_BIN       = '/home/giuliano/git/cuda/fragmentation/data/lodi1954.tif';
% FIL_ROI		= '/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels_roi.tif';
% FIL_BIN		= '/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels.tif';
% FIL_ROI       = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif';
% FIL_BIN       = '/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif';
% FIL_ROI		= '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2_roi.tif';
% FIL_BIN		= '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2.tif';
% FIL_ROI		= '/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped_roi.tif';
% FIL_BIN		= '/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped.tif';
% FIL_ROI		= '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006.tif';
% FIL_BIN		= '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006.tif';

cpy_ssgci_data  = 0; % Do you want to copy MapStore data and run check on these?
cu_compile      = 0; % Do you want MatLab compile your source .cu file?
cu_run          = 1; % Do you want to run the cuda code (compiled here or outside)?
save_mats       = 0; % Do you want to store BIN & MAT files for specific runs?
print_me        = 1; % Do you want .cu code print the output of single every kernel?
create_bin      = 0; % Do you want to create a new BIN array for a new comparison?
deep_check      = 1; % Do you want to perform a deep comparison?
plot_me         = 1; % Do you want to plot following maps? {bin,roi,Lml,Lcuda}
% BIN characteristics:
% tile size in XY – fixed [no more then 32x32]:
tiledimX        = 32;  % 512 - 32 - 30
tiledimY        = 32;  % 2   - 32 - 30
% number of tiles in XY:
ntilesX         = 13;
ntilesY         = 12;
threshold       = 0.8300; % set image sparsity: near 0 high density, near 1 high sparsity
%% set dim:
if create_bin
% NOTE:
%   Here it is assumed that the image can be partitioned into tiles having
%   1-pixel-width extra border along the tile perimeter. Therefore the
%   tiledimX is the length of the tile in X geographical dimension
%   including the extra border. Hence in order to be straight in sizing the
%   image, I fistly fix the tile dimensions (and the number of tiles, to
%   set the total size of the overall image) and then the number of rows
%   and columns is derived accordingly.

    % % fixed [no more then 32x32]:
    % tiledimX    = 32;  % 512 - 32 - 30
    % tiledimY    = 32;  % 2   - 32 - 30

    % consequence:
    NC          = (ntilesX-1)*(tiledimX-1) + tiledimX -2;
    NR          = (ntilesY-1)*(tiledimY-1) + tiledimY -2;
end
%% arguments to run within Nsight
% fprintf('%d %d %d %d %d\n',tiledimX,tiledimY,NC,NR,print_me);
%% create BIN & ROI
if create_bin
    fprintf('Creating new BIN & ROI...\t')
    
    % BIN:
    BIN                     = rand(NR,NC);
    BIN(BIN>=(1-threshold)) = 1;
    BIN(BIN< (1-threshold)) = 0;
    BIN                     = logical(BIN);
    
    % ROI:
    % -- random:
    fprintf('  ROI has random ones on map.')
    ROI                     = rand(NR,NC);
    ROI                     = imfilter(ROI,ones(3)/9);
    ROI(ROI>=(1-0.7))       = 1;
    ROI(ROI< (1-0.7))       = 0;
    ROI = logical(ROI);
    
    % -- all ones:
%     fprintf('  ROI is all one.')
%     ROI                     = true(size(BIN));
    
    % modification to test for irregular NC*NR
%     NC = NC-2;
%     NR = NR-3;
%     BIN = BIN(1:end-3,1:end-2);
    fprintf('...done!\n')
end
%% save BIN & ROI as geotiff
if create_bin
    fprintf('Saving new BIN & ROI...\t')
    
    % Build the new georeferenced GRID:
    info                    = geotiffinfo( fullfile('/home/giuliano/git/cuda/fragmentation/data','BIN.tif') );
    R                       = info.SpatialRef;
    newXlim                 = [ R.XLimWorld(1), R.XLimWorld(1) + NC*R.DeltaX ];
    newYlim                 = [ R.YLimWorld(1), R.YLimWorld(1) + NR*(-R.DeltaY) ];
    Rnew                    = maprasterref( ...
                'XLimWorld',              newXlim, ...
                'YLimWorld',              newYlim, ...
                'RasterSize',             [NR,NC], ...
                'RasterInterpretation',   R.RasterInterpretation, ...
                'ColumnsStartFrom',       R.ColumnsStartFrom, ...
                'RowsStartFrom',          R.RowsStartFrom ...
                                      );
    geotiffwrite( FIL_BIN,   BIN, Rnew, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag );
    geotiffwrite( FIL_ROI,   ROI, Rnew, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag );
    if ~cu_run% if I use cu_run=true I don't need to manually run the CUDA-C program, it is done from within MatLab.
        fprintf('MatLab saved the following Input maps to be used from within CUDA-C:\n\t-%s\n\t-%s\n', ...
              FIL_BIN, FIL_ROI)
        fprintf('The execution is now paused!\nRun CUDA-C code using the maps given above...\t(press a key after doing that!)\n')
        pause
    end
    fprintf('...done!\n')
end
%% crop input GRID to fit CUDA blockSize(32,32)
% if create_bin
% % BIN
% ntiles              = floor( size(BIN) ./ [tiledimY,tiledimX] );
% newSize             = [186,280];
% [path,name,ext]     = fileparts( FIL_BIN );
% FIL_BIN_crop        = sprintf('%s/%s%s%s',path,name,'-cropped',ext);
% gclipgrid( FIL_BIN, FIL_BIN_crop, newSize );
% FIL_BIN = FIL_BIN_crop;
% end
%% load BIN & ROI from geotiff
force_to_load = true;
if ~create_bin || force_to_load
    if cpy_ssgci_data
        try
            copyfile('/opt/soil_sealing/exchange_data/testing/ccl_1toN_hist_*', fullfile(BASE_DIR,'data') )
        catch exception
            warning('%s ––> %s\n',exception.identifier,exception.message)
        end
    end
    fprintf('Loading BIN... %s\t',FIL_BIN)
    BIN         = geotiffread( FIL_BIN );
    fprintf('...done!\n')
    fprintf('Loading ROI... %s\t',FIL_ROI)
    ROI         = geotiffread( FIL_ROI );
    fprintf('...done!\n')
end
%% set derived variables
WIDTH           = size(BIN,2);
HEIGHT          = size(BIN,1);
myToc           = zeros(1,1); % computing time [msec]
ntilesX         = ceil( (WIDTH+2-1)  / (tiledimX-1)  );
ntilesY         = ceil( (HEIGHT+2-1) / (tiledimY-1)  );
%% size on GPU
% size of BIN: {BIN,ROI}
SIZE__(1)   = HEIGHT*WIDTH*2 *1;% 8 bits = 1 bytes
% size of LAB-MAT: {lab_mat_gpu, lab_mat_gpu_f, lab_mat_gpu_1N, +1 to account other stuff}
SIZE__(2)   = ntilesX*ntilesY*tiledimX*tiledimY*4 *4;% 32 bits = 4 bytes
% print:
% fprintf('Size on GPU-memory:\t%.1f MB\n', sum( SIZE__ )/10^6)
%% compute MatLab labeled image & hist
fprintf('Computing MatLab CCL & histogram...\t')

hist_thresh = 10;
tic,
BINf        = BIN;
% BINf(BINf==0 | ROI==0) = 0;
% BINf(BINf==1 & ROI==1) = 1;
BINf(ROI==0)= 0;

Lml         = transpose( bwlabel(BINf',8) );
IDml        = unique(Lml); if IDml(1)==0, IDml(1)=[]; end
hist_ml     = zeros(size(IDml));
T(1)        = toc;

% N = histcounts(X,nbins) 
tic
for ii = 1: min(hist_thresh,numel(IDml))
    hist_ml(ii) = sum(Lml(:)==IDml(ii));
end
tmp         = toc;
if hist_thresh < numel(IDml)
    T(1)    = T(1) + tmp*numel(IDml)/hist_thresh;
else
    T(1)    = T(1) + tmp;
end

fprintf('...done!\n')
%% save MatLab labeled image
% save(fullfile(BASE_DIR,'data',WRITE_TXT('LAB_MAT-ml-',ntilesX,ntilesY,tiledimX,tiledimY)), 'BIN', '-ascii')
if save_mats
    dlmwrite(fullfile(BASE_DIR,'data',WRITE_TXT('LAB_MAT-ml-',ntilesX,ntilesY,tiledimX,tiledimY)),BIN,'delimiter',' ')
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
    setenv('LD_LIBRARY_PATH', '');
    unix( ['sh ',fullfile(BASE_DIR,'matlab','compile-fil')] );
end
% % OLD
% comp_pars_1 = '--compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -x cu -o';
% comp_pars_2 = '--cudart static -link -o';
%% execute CUDA-code & grab stout
if cu_run
    
    setenv('LD_LIBRARY_PATH', '');
    
    % run_pars        = [' ',num2str(tiledimX),' ',num2str(tiledimY),' ',num2str(WIDTH),' ',num2str(HEIGHT),' ',num2str(print_me)];
    run_pars        = '';
    run_str         = [exefil, run_pars];
    fprintf('\n%s\n',repmat('–',1,130))
    fprintf('Running CUDA-C program:\n\t%s\n',run_str);
    fprintf('%s\n',repmat('–',1,130))
    tic
    [status,out]    = unix( run_str, '-echo' );
    T(3)            = toc;
    fprintf('%s\n\n\n',repmat('–',1,130))
    sq_open         = strfind(out,'[');
    str_divide      = strfind(out,':');
    T(2)            = str2double( out(str_divide(end)+1:sq_open(end) -1) )/1000;
end
%% import CUDA labeled image & hist
fprintf('Importing CUDA CCL & histogram...\n')
if exist( FIL_LAB, 'file')
    Lcuda       = double( geotiffread( FIL_LAB ) );
    if exist( FIL_LABrand, 'file')
        Lcuda_rand  = double( geotiffread( FIL_LABrand ) );
    end
else
    Lcuda       = double( geotiffread( FIL_LABrand ) );
    warning('Cuda 1-to-N labeled image not found [%s]\nImporting the random labeled image: [%s]',FIL_LAB, FIL_LABrand)
end

if exist( FIL_HIST, 'file')
    hist_cu     = load( FIL_HIST );
    hist_cu(1)  = [];
    cuda_relabel = true;
else
    cuda_relabel = false;
    warning('Cuda histogram file not found! [%s]',FIL_HIST)
end

fprintf('...done!\n')
%% import CUDA labeled image [in itinere, by hand]
% % right click on file "-4-intra_tile_re_label.txt" --> Import data... -->
% % set parameters as required.
% % Then, run the following before comparison:
% Lcuda = intratilerelabel(2:end,:);
% % remove duplicated LINES (rows & cols):
% Lcuda(:,tiledimX:tiledimX:size(Lcuda,2)-1)=[];
% Lcuda(tiledimY:tiledimY:size(Lcuda,1)-1,:)=[];
%% store images
if save_mats
    while ~exist( fullfile(BASE_DIR,'data','CUDA-code.txt'), 'file' ), end
    copyfile(fullfile(BASE_DIR,'data',FIL_BIN),fullfile(BASE_DIR,'data',WRITE_TXT('BIN_MAT',ntilesX,ntilesY,tiledimX,tiledimY)))
    movefile(fullfile(BASE_DIR,'data','CUDA-code.txt'),fullfile(BASE_DIR,'data',WRITE_TXT('LAB_MAT-cu-',ntilesX,ntilesY,tiledimX,tiledimY)))
end
%% [1-to-N]
%% -- compare | labels [1toN vs ml]
fprintf('\nC O M P A R I S O N   o f   L A B E L S :: [1toN vs ml]\n')
DIFFER  = Lcuda - Lml;
idx     = find(DIFFER);
UL      = unique([Lml(idx),Lcuda(idx)],'rows');
UC      = unique([Lcuda(idx),Lml(idx)],'rows');
ucu     = unique(Lcuda(:));
if ucu(1)==0, ucu(1)=[]; end
nIDs    = numel(ucu);

fprintf( '%s\n', repmat('–',1,65) );
fprintf( '   The Connected Component Labeling (CCL) procedure\n' );
fprintf( '\t       # CUDA vs MatLab # \n' );
fprintf( '   Image[%d,%d], tiledim[%d,%d], ntiles[%d,%d] \n',HEIGHT,WIDTH,tiledimX,tiledimY,ntilesX,ntilesY );
fprintf( '   Numel[%d], NObjPix[%d], NBackgroundPix[%d]\n',numel(BIN),sum(BIN(:)),sum(BIN(:)==0));
fprintf( '   Size on GPU memory:\t%.1f MB\n', sum( SIZE__ )/10^6)
% fprintf( '   differs in <<%d>> cells [norm=%.2f]\n', length(find(DIFFER)),norm(DIFFER) );
fprintf( '     > nID: %d (%s) vs %d (%s)\n',nIDs,'cuda',max(Lml(:)),'matlab' );
% if isempty(UL) Nml = 0; else Nml=sum(diff(UL(:,1))) ~= size(UL,1)-1; end
if isempty(UL) Nml = 0; else Nml=sum(diff(UL(:,1))==0); end
fprintf( '     > %d MatLab labels have more CUDA labels\n', Nml );
% if isempty(UC) Ncu = 0; else Ncu=sum(bwlabel(diff(UC(:,1)),4)) ~= size(UC,1)-1; end
if isempty(UC) Ncu = 0; else Ncu=sum(diff(UC(:,1))==0); end
fprintf( '     > %d CUDA labels have more MatLab labels\n', Ncu );
fprintf( '     > speed-up (ml/cu) = %3.1f\t( %3.2f / %3.2f [ms] )\n', T(1)/T(2),T(1)*1000,T(2)*1000 );
fprintf( '%s\n\n', repmat('–',1,65) );
%% -- find problems |  [1toN vs ml]
fprintf('Searching for problems...\n')
eFound = unique( UL( find(diff(UL(:,1))==0), 1 ) );

for ii = 1:min(20,length(eFound))
    fprintf('  MatLab ID[ %4d ] ---> [',eFound(ii));
    List = UL(UL(:,1)==eFound(ii),:);
    for jj=1:sum(UL(:,1)==eFound(ii))
        fprintf('%7d, ',List(jj,2) );
    end
    fprintf(']\n');
end
if 20<length(eFound)
    fprintf('  ...\n')
end
fprintf('...end!\n')
%% [random]
%% -- compare | labels [rand vs ml]
fprintf('\nC O M P A R I S O N   o f   L A B E L S :: [rand vs ml]\n')
DIFFER  = Lcuda_rand - Lml;
idx     = find(DIFFER);
UL      = unique([Lml(idx),Lcuda_rand(idx)],'rows');
UC      = unique([Lcuda_rand(idx),Lml(idx)],'rows');
ucu     = unique(Lcuda_rand(:));
if ucu(1)==0, ucu(1)=[]; end
nIDs    = numel(ucu);

fprintf( '%s\n', repmat('–',1,65) );
fprintf( '   The Connected Component Labeling (CCL) procedure\n' );
fprintf( '\t       # CUDA vs MatLab # \n' );
fprintf( '   Image[%d,%d], tiledim[%d,%d], ntiles[%d,%d] \n',HEIGHT,WIDTH,tiledimX,tiledimY,ntilesX,ntilesY );
fprintf( '   Numel[%d], NObjPix[%d], NBackgroundPix[%d]\n',numel(BIN),sum(BIN(:)),sum(BIN(:)==0));
fprintf( '   Size on GPU memory:\t%.1f MB\n', sum( SIZE__ )/10^6)
% fprintf( '   differs in <<%d>> cells [norm=%.2f]\n', length(find(DIFFER)),norm(DIFFER) );
fprintf( '     > nID: %d (%s) vs %d (%s)\n',nIDs,'cuda',max(Lml(:)),'matlab' );
% if isempty(UL) Nml = 0; else Nml=sum(diff(UL(:,1))) ~= size(UL,1)-1; end
if isempty(UL) Nml = 0; else Nml=sum(diff(UL(:,1))==0); end
fprintf( '     > %d MatLab labels have more CUDA labels\n', Nml );
% if isempty(UC) Ncu = 0; else Ncu=sum(bwlabel(diff(UC(:,1)),4)) ~= size(UC,1)-1; end
if isempty(UC) Ncu = 0; else Ncu=sum(diff(UC(:,1))==0); end
fprintf( '     > %d CUDA labels have more MatLab labels\n', Ncu );
fprintf( '     > speed-up (ml/cu) = %3.1f\t( %3.2f / %3.2f [ms] )\n', T(1)/T(2),T(1)*1000,T(2)*1000 );
fprintf( '%s\n\n', repmat('–',1,65) );
%% -- find problems |  [rand vs ml]
fprintf('Searching for problems...\n')
eFound = unique( UL( find(diff(UL(:,1))==0), 1 ) );

for ii = 1:min(20,length(eFound))
    fprintf('  MatLab ID[ %4d ] ---> [',eFound(ii));
    List = UL(UL(:,1)==eFound(ii),:);
    for jj=1:sum(UL(:,1)==eFound(ii))
        fprintf('%7d, ',List(jj,2) );
    end
    fprintf(']\n');
end
if 20<length(eFound)
    fprintf('  ...\n')
end
fprintf('...end!\n')
%% [SS-GCI]
%% -- compare | labels [ss-gci vs 1toN]
fprintf('\nC O M P A R I S O N ::  s s – g c i   l a b e l s\n')
if exist('FIL_LAB_ssgci','var') && cpy_ssgci_data
    Lcuda_ssgci  = double( geotiffread( FIL_LAB_ssgci ) );
    DIFFER  = Lcuda_ssgci - Lcuda;
    idx     = find(DIFFER);
    UL      = unique([Lcuda(idx),Lcuda_ssgci(idx)],'rows');
    UC      = unique([Lcuda_ssgci(idx),Lcuda(idx)],'rows');
    ucu     = unique(Lcuda_ssgci(:));
    if ucu(1)==0, ucu(1)=[]; end
    u1tn     = unique(Lcuda(:));
    if u1tn(1)==0, u1tn(1)=[]; end
    nIDs    = numel(ucu);
    nID1tn  = numel(u1tn);
% SUBSTITUTIONS :: ml ––> 1toN, cuda ––> ss-gci
    fprintf( '%s\n', repmat('–',1,65) );
    fprintf( '   The Connected Component Labeling (CCL) procedure\n' );
    fprintf( '\t       # SS-GCI vs 1toN # \n' );
    fprintf( '   Image[%d,%d], tiledim[%d,%d], ntiles[%d,%d] \n',HEIGHT,WIDTH,tiledimX,tiledimY,ntilesX,ntilesY );
    fprintf( '   Numel[%d], NObjPix[%d], NBackgroundPix[%d]\n',numel(BIN),sum(BIN(:)),sum(BIN(:)==0));
    fprintf( '   Size on GPU memory:\t%.1f MB\n', sum( SIZE__ )/10^6)
    % fprintf( '   differs in <<%d>> cells [norm=%.2f]\n', length(find(DIFFER)),norm(DIFFER) );
    fprintf( '     > nID: %d (%s) vs %d (%s)\n',nIDs,'ss-gci',nID1tn,'1toN' );
    % if isempty(UL) Nml = 0; else Nml=sum(diff(UL(:,1))) ~= size(UL,1)-1; end
    if isempty(UL) Nml = 0; else Nml=sum(diff(UL(:,1))==0); end
    fprintf( '     > %d 1toN labels have more ss-gci labels\n', Nml );
    % if isempty(UC) Ncu = 0; else Ncu=sum(bwlabel(diff(UC(:,1)),4)) ~= size(UC,1)-1; end
    if isempty(UC) Ncu = 0; else Ncu=sum(diff(UC(:,1))==0); end
    fprintf( '     > %d ss-gci labels have more 1toN labels\n', Ncu );
    fprintf( '     > speed-up (1toN/ss-gci) = %3.1f\t( %3.2f / %3.2f [ms] )\n', T(1)/NaN,T(1)*1000,NaN );
    fprintf( '%s\n\n', repmat('–',1,65) );
else
    fprintf( '%s\n', '...the file produced on-the-fly over the SS-GCI is not available!' )
end
fprintf('End!\n')
%% -- find problems |  [ss-gci vs 1toN]
fprintf('Searching for problems...\n')
if exist('FIL_LAB_ssgci','var') && cpy_ssgci_data
    eFound = unique( UL( find(diff(UL(:,1))==0), 1 ) );
% SUBSTITUTIONS :: ml ––> 1toN, cuda ––> ss-gci
    for ii = 1:min(20,length(eFound))
        fprintf('  1toN ID[ %4d ] ---> [',eFound(ii));
        List = UL(UL(:,1)==eFound(ii),:);
        for jj=1:sum(UL(:,1)==eFound(ii))
            fprintf('%7d, ',List(jj,2) );
        end
        fprintf(']\n');
    end
    if 20<length(eFound)
        fprintf('  ...\n')
    end
else
    fprintf( '%s\n', '...the file produced on-the-fly over the SS-GCI is not available!' )
end
fprintf('...end!\n')
%% compare | histogram
fprintf('\nC O M P A R I S O N   o f   H I S T O G R A M  ::  [1toN vs ml] \n')
if cuda_relabel
   
% ** SELECTOR FOR PRINT AMOUNT ––> 1 or 2? **
% 1 % to limit the number of prints:
%     Nids        = 10;
%     Nids        = min(Nids,numel(IDml));
%     kind_chk    = fprintf('  [The checking procedure is incomplete!]\n  [Set option #2 for full checking the histogram counts!]\n');
% 2 % to print all objects with step 100
    Nids        = numel(IDml);
    kind_chk    = fprintf('  [A full checking procedure is selected!]\n  [This is time-consuming for large histograms!]\n');
% ** SELECTOR FOR PRINT AMOUNT ––> 1 or 2? **
    
    printed_line    = 0;
    Cerr__          = 0;
    hist_ml         = zeros(size(IDml));
    fprintf( '%s\n', repmat('–',1,65) );
    % MatLab is my truth: Ids to be Checked:
    Cids = randperm(numel(hist_ml),Nids);% Cids is the list of MatLab IDs
    fprintf('%6s%20s%20s%18s\n','o b j','l a b e l s','h i s t o g r a m','e r r o r')
    fprintf('%6s%10s%10s%10s%10s%18s\n','#','CUDA','MatLab','CUDA','MatLab','cuda vs matlab')
    for ii = 1:Nids
        %if Cerr__ > 20 || ii>20, break, end %activate this to limit the number of prints
        err__ = false;
        hist_ml(Cids(ii)) = sum(Lml(:)==Cids(ii));
        [ridx,cidx] = find(Lml==Cids(ii),1,'first');
        currIdxCu = Lcuda(ridx,cidx);
        if currIdxCu==0
            printed_line = printed_line +1;
            err__ = true;
            Cerr__ = Cerr__ +1;
            fprintf('%6d%10d%10d%10d%10d%17d*\n',ii,currIdxCu,Cids(ii),currIdxCu,hist_ml(Cids(ii)),err__)
        elseif hist_ml(Cids(ii)) ~= hist_cu(currIdxCu)
            printed_line = printed_line +1;
            err__ = true;
            Cerr__ = Cerr__ +1;
            fprintf('%6d%10d%10d%10d%10d%17d*\n',ii,currIdxCu,Cids(ii),hist_cu(currIdxCu),hist_ml(Cids(ii)),err__)
        else
            if printed_line>8 && ~mod(ii,10)==0, continue, end % do not write good ones always but every 100 steps
            printed_line = printed_line +1;
            fprintf('%6d%10d%10d%10d%10d%18d\n',ii,currIdxCu,Cids(ii),hist_cu(currIdxCu),hist_ml(Cids(ii)),err__)
        end
    end
    if Nids<numel(IDml)
        [~,~,iIDml]=setxor(Cids,IDml);
        fprintf(' . . .\n')
        err__ = false;
        hist_ml(iIDml(end)) = sum(Lml(:)==iIDml(end));
        [ridx,cidx] = find(Lml==iIDml(end),1,'first');
        currIdxCu = Lcuda(ridx,cidx);
        if hist_ml(iIDml(end)) ~= hist_cu(currIdxCu)
            err__ = true;
            fprintf('%6d%10d%10d%10d%10d%17d*\n',numel(IDml),currIdxCu,iIDml(end),hist_cu(currIdxCu),hist_ml(iIDml(end)),err__)
        else
            fprintf('%6d%10d%10d%10d%10d%18d\n',numel(IDml),currIdxCu,iIDml(end),hist_cu(currIdxCu),hist_ml(iIDml(end)),err__)
        end        
    end
    
    fprintf( '%s\n', repmat('–',1,65) );
    if Cerr__>0
        fprintf('%45s%18d\n','Errors count is more than ––>',Cerr__);
    elseif ii==Nids
        fprintf('%35s[%5d] ––>%18d\n','Errors count on all objects',Cerr__,ii);
    elseif ii>20
        fprintf('%45s%18d\n','Errors count till object 20 ––>',Cerr__);
    end
else
    warning('Impossible without the file %s',FIL_HIST);
end
fprintf( '\n' )
%% compare | labels | IDs to maps [1-to-N vs random]
if exist('Lcuda_rand','var') %&& false
    printed_line = 0;
    
    ID_1toN = load(FIL_ID1N);% linked to Lcuda
    ID_rand = load(FIL_IDra);% linked to Lcuda_rand
    
    fprintf('\nC O M P A R I S O N :: b i n s  |  1–to–N vs random \n')
    Nids        = numel(ID_1toN);
    hist_cu_1N  = zeros(size(ID_1toN));
    hist_cu_ra  = zeros(size(ID_1toN));

    fprintf( '%s\n', repmat('–',1,65) );
    fprintf('%6s%20s%20s%18s\n','o b j','l a b e l s','h i s t o g r a m','e r r o r')
    fprintf('%6s%10s%10s%10s%10s%18s\n','','ID_rand','ID_1toN','map[cu]','map[ra]','')
    fprintf('%6s%10s%10s%10s%10s%18s\n','#','random','1–to-N','random','1–to-N','1–to-N vs random')
    for ii = 1:Nids
        if printed_line >8, continue, end
        err__           = false;
        Fcu_rand_ii = Lcuda_rand==ID_rand(ii);
        Fcu_1toN_ii = Lcuda     ==ID_1toN(ii);
        hist_cu_ra(ii)  = sum(Fcu_rand_ii(:));
        hist_cu_1N(ii)  = sum(Fcu_1toN_ii(:));

        % print only errors
        if hist_cu_1N(ii) ~= hist_cu_ra(ii)
            err__ = true;
            fprintf('%6d%10d%10d%10d%10d%17d*\n',ii,ID_rand(ii),ID_1toN(ii),hist_cu_ra(ii),hist_cu_1N(ii),err__)
            printed_line = printed_line +1;
        else
%             fprintf('%6d%10d%10d%10d%10d%18d\n',ii,currIdxCu,Cids(ii),hist_cu_ra(currIdxCu),hist_cu_1N(Cids(ii)),err__)
        end
    end
    if printed_line>8, fprintf('  ...\n'), end
    fprintf( '%s\n', repmat('–',1,65) );

    obj_ml      = Lml>0;
    obj_cura    = Lcuda_rand>0;
    obj_cu1N    = Lcuda>0;
    Dobj_ra     = sum(sum( obj_ml - obj_cura ));
    Dobj_1N     = sum(sum( obj_ml - obj_cu1N ));
    fprintf( '%26s%10d%10d%18d\n','TOT.',sum(obj_cura(:)),sum(obj_cu1N(:)), sum(obj_cura(:)-obj_cu1N(:)) );
    fprintf( '%26s%10d%10d%18d\n','[using ml]  DIFF.',Dobj_ra,Dobj_1N, Dobj_ra-Dobj_1N);
    fprintf( '\n')
end
%% check number of pixels are the same (without accounting for objects)
% if deep_check
    fprintf('Unrecognized pixels...\n');
    Lml_bin = Lml;
    Lml_bin(Lml >= 1) = 1;
    fprintf('Number of unrecognized object pixels: \t%10s%10d\n','[MatLab]',sum( double(BIN(:)).*double(ROI(:)) -Lml_bin(:)) );

    Lcuda_bin = Lcuda;
    Lcuda_bin(Lcuda >= 1) = 1;
    fprintf('Number of unrecognized object pixels: \t%10s%10d\n','[CUDA]',sum( double(BIN(:)).*double(ROI(:)) -Lcuda_bin(:))  );
    fprintf('...end\n')
% end
%% find unrecognized pixels
DIFF    = double(BIN).*double(ROI) -Lcuda_bin;
[r,c]   = find(DIFF);
%% [deep] compare | histogram | object-by-object | maps [1toN vs ml]
fprintf('\nD E E P   C O M P A R I S O N :: m a p s  | 1toN vs ml\n')
if deep_check

    Ucu = unique(Lcuda(:));
    Uml = unique(Lml(:));
    % delete the background pixels from bins:
    Ucu(1) = [];
    Uml(1) = [];
    
    % first check :: labels are in range [1,N]
    fprintf('  Check [on MAPS] that labels are in range [1,N]\n')
    % -a- about N
    Ncu = max(Ucu)==length(Ucu);
    Nml = max(Uml)==length(Uml);
    fprintf('    -a- about N :: ml[valid=%d], cu[valid=%d]\n',Nml,Ncu)
    % -b- about sequence
    Ncu = sum(diff(Ucu)>1);
    Nml = sum(diff(Uml)>1);
    fprintf('    -b- about sequence :: ml[discontinuities=%d], cu[discontinuities=%d]\n',Nml,Ncu)

    fprintf('    -c- list of wrong correspondences:\n')
    parfor ii = 1:numel(Ucu)
        Fcu     = Lcuda==Ucu(ii);
        [r,c]   = find(Fcu,1,'first');
        IDml    = Lml(r,c);
        Fml     = Lml==IDml;
        % compare the same object between Cuda and Matlab:
        isOne   = sum(abs( Fcu(:) - Fml(:) ));

        if isOne
            fprintf( '      %5d, cu[ID=%10d,No=%4d], ml[ID=%4d,No=%4d]\n',ii,Ucu(ii),sum(Fcu(:)),IDml,sum(Fml(:)) );
        else
%             fprintf('%d\n',ii)
        end
    end
    fprintf('    -c- End\n')
else
    fprintf('  ...disabled!\n')
end
fprintf('Finished!\n')
fprintf( '\n' )
%% [deep] compare | histogram | object-by-object | maps [1toN vs rand]
fprintf('\nD E E P   C O M P A R I S O N :: m a p s  | 1toN vs rand\n')
if deep_check
    Ucu = unique(Lcuda(:));
    Ura = unique(Lcuda_rand(:));
    % delete the background pixels from bins:
    Ucu(1) = [];
    Ura(1) = [];

    parfor ii = 1:numel(Ucu)
        Fcu     = Lcuda==Ucu(ii);
        [r,c]   = find(Fcu,1,'first');
        IDra    = Lcuda_rand(r,c);
        Fra     = Lcuda_rand==IDra;
        % compare the same object between Cuda and Matlab:
        isOne   = sum(abs( Fcu(:) - Fra(:) ));

        if isOne
            fprintf( '%5d, cu[ID=%10d,No=%4d], ml[ID=%4d,No=%4d]\n',ii,Ucu(ii),sum(Fcu(:)),IDra,sum(Fra(:)) );
        else
%             fprintf('%d\n',ii)
        end
    end
    fprintf('  Finished!\n')
else
    fprintf('...disabled!\n')
end
fprintf( '\n' )
%% plot
if plot_me
    % figure(1),subplot(131),gpcolor(DIFFER),title('(CUDA - MatLab) CCL differences')
    % figure(1),subplot(132),gpcolor(Lml),title('MatLab')
    % figure(1),subplot(133),gpcolor(Lcuda),title('CUDA')
    figure(11)
    subplot(221),imshow(logical(BIN)),title('BIN'),pause(0.5)
    subplot(222),imshow(logical(ROI)),title('ROI'),pause(0.5)
    subplot(223),imshow(label2rgb(Lcuda)),title('CUDA'),pause(0.5)
    subplot(224),imshow(label2rgb(Lml)),title('MatLab'),pause(0.5)
end
%% speedup
% fprintf('\n');
% fprintf('%8s%10s: %7.1f [ms]\n', 'Speed','[MatLab]', T(1)*1000 );
% fprintf('%8s%10s: %7.1f [ms]\n', 'Speed','[CUDA]', T(2)*1000 );
% fprintf('%8s%10s: %7.1f [--]\n', 'Speedup','[ml/cu]',T(1) / T(2) );
%% test kerneles step-by-step
fprintf('\nT E S T   S I N G L E   K E R N E L S\n')
if print_me
    kern_0		= 'filter_roi';
    kern_1      = 'intra_tile_labeling';
    kern_2      = 'stitching_tiles';
    kern_3      = 'root_equivalence';
    kern_4      = 'intra_tile_re_label';
    kern_4_a 	= 'count_labels';
    kern_4_b 	= 'labels__1_to_N';
    kern_4_c 	= 'intratile_relabel_1toN';
    kern_5      = 'del_duplicated_lines';
    kern_6      = 'histogram';

    % ROI * BIN:
    G_0         = geotiffread( fullfile(BASE_DIR,'data',['-0-',kern_0,'.tif']) );
    % Random CCL
    G_1         = geotiffread( fullfile(BASE_DIR,'data',['-1-',kern_1,'.tif']) );
    G_2         = geotiffread( fullfile(BASE_DIR,'data',['-2-',kern_2,'.tif']) );
    G_3         = geotiffread( fullfile(BASE_DIR,'data',['-3-',kern_3,'.tif']) );
    G_4         = geotiffread( fullfile(BASE_DIR,'data',['-4-',kern_4,'.tif']) );
    % 1-to-N CLL
    G_5         = load( fullfile(BASE_DIR,'data',       ['-5-',kern_4_a,'.txt']) );
    G_6         = geotiffread( fullfile(BASE_DIR,'data',['-6-',kern_4_b,'.tif']) );
    G_7         = geotiffread( fullfile(BASE_DIR,'data',['-7-',kern_4_c,'.tif']) );

    % G{8}        = geotiffread( fullfile(BASE_DIR,'data',['-1-',kern_1,'.tif']) );
    % G{9}        = geotiffread( fullfile(BASE_DIR,'data',['-1-',kern_1,'.tif']) );

    % Proof that "filter_roi" kernel works fine:
    DIFF    = double(G_0) - double(BINf);
    ERR     = sum(sum( abs( DIFF ) ));
    [r0,c0]   = find(DIFF);
    fprintf('|%02d|%25s : err pixels[%d]\n\n',0,kern_0,numel(r0))
    if plot_me
        figure(27)
        subplot(221),imshow(logical(BINf)),title('BIN - ml'),pause(1.5)
        subplot(222),imshow(logical(G_0)),title(['BIN - ',kern_0])
        subplot(223),imshow(logical(ROI)),title('ROI')
        subplot(224),imshow(logical(DIFF)),title('DIFF')
    end

    % Proof that output from "intra_tile_re_label" is the same of Cuda random:
    % DIFF_4  = double(G_4) - Lcuda_rand;
    % ERR_4   = sum(sum( DIFF_4 ));
    % [r4,c4]   = find(DIFF_4);
    fprintf('|%02d|%25s : err pixels[%d]\n',4,kern_4,0)
    Uml = unique(Lml); if Uml(1)==0; Uml(1)=[]; end
    Count = 0;
    for ii = 1:numel(Uml)
        % -a- fix the label in ml:
        idx = find( Lml==Uml(ii) );
        % -b- extract labels from G_4 with same map location:
        Fg4 = G_4(idx);
        % -c- check that G_4 labels are all identical
        Ug4 = unique(Fg4);
        if numel(Ug4)>1
            Count = Count +1;
            fprintf('    |ID| ml[%3d] ––> G_4{ %d',Uml(ii),Ug4(1))
            for jj = 2:numel(Ug4)
                fprintf(', %d', Ug4(jj))
            end
            fprintf(' }\n')
        end
    end
    if Count >0 fprintf('|%02d|%25s : err pixels[%d]\n\n',4,kern_4,Count), end
else
    fprintf('...disabled!\n')
end
fprintf( '\n' )
