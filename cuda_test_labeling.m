%% NEW VERS of CCL WITH SHARED-MEM
clear,clc
BASE_DIR        = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing';
T               = NaN(2,1);
% –/O
FIL_LAB         = '/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/LAB-MAT-cuda.tif';
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
FIL_ROI         = fullfile(BASE_DIR,'data','created-on-the-fly_ROI.tif');
FIL_BIN         = fullfile(BASE_DIR,'data','created-on-the-fly_BIN.tif');
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

cu_compile      = 0; % Do you want MatLab compile your source .cu file?
cu_run          = 1; % Do you want to run the cuda code (compiled here or outside)?
save_mats       = 0; % Do you want to store BIN & MAT files for specific runs?
print_me        = 0; % Do you want .cu code print LAB-MAT at the end of every kernel?
create_bin      = 1; % Do you want to create a new BIN array for a new comparison?
deep_check      = 0; % Do you want to perform a deep comparison?
% BIN characteristics:
% tile size in XY – fixed [no more then 32x32]:
tiledimX        = 32;  % 512 - 32 - 30
tiledimY        = 32;  % 2   - 32 - 30
% number of tiles in XY:
ntilesX         = 435;
ntilesY         = 383;
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
    
    BIN                     = rand(NR,NC);
    BIN(BIN>=(1-threshold)) = 1;
    BIN(BIN<(1-threshold))  = 0;
    BIN                     = logical(BIN);
    ROI                     = true(size(BIN));
    % modification to test for irregular NC*NR
    % NC = NC-2;
    % NR = NR-3;
    % BIN = BIN(1:end-3,1:end-2);
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
if ~create_bin
    fprintf('Loading BIN... %s\t',FIL_BIN)
    BIN         = geotiffread( FIL_BIN );
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
Lml         = double(ROI) .* transpose( bwlabel(BIN',8) );
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
%% -- compare | labels [1-to-N]
fprintf('\nC O M P A R I S O N :: 1-to-N   l a b e l s\n')
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
%% -- find problems |  [1-to-N]
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
%% -- compare | labels [random]
fprintf('\nC O M P A R I S O N :: r a n d o m   l a b e l s\n')
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
%% -- find problems |  [1-to-N]
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
%% compare | histogram
fprintf('\nC O M P A R I S O N :: h i s t o g r a m \n')
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
        if Cerr__ > 20 || ii>20, break, end
        err__ = false;
        hist_ml(Cids(ii)) = sum(Lml(:)==Cids(ii));
        [ridx,cidx] = find(Lml==Cids(ii),1,'first');
        currIdxCu = Lcuda(ridx,cidx);
        if hist_ml(Cids(ii)) ~= hist_cu(currIdxCu)
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
    elseif ii>20
        fprintf('%45s%18d\n','Errors count till object 20 ––>',Cerr__);
    end
else
    warning('Impossible without the file %s',FIL_HIST);
end
fprintf( '\n' )
%% compare | labels [1-to-N vs random]
if exist('Lcuda_rand','var') && false
    printed_line = 0;
    
    ID_1toN = load(FIL_ID1N);% linked to Lcuda
    ID_rand = load(FIL_IDra);% linked to Lcuda_rand
    
    fprintf('\nC U D A   C O M P A R I S O N :: 1–to–N vs random \n')
    Nids        = numel(ID_1toN);
    hist_cu_1N  = zeros(size(ID_1toN));
    hist_cu_ra  = zeros(size(ID_1toN));

    fprintf( '%s\n', repmat('–',1,65) );
    fprintf('%6s%20s%20s%18s\n','o b j','l a b e l s','h i s t o g r a m','e r r o r')
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
    fprintf('Number of unrecognized object pixels: \t%10s%10d\n','[MatLab]',sum( double(BIN(:))-Lml_bin(:)) );

    Lcuda_bin = Lcuda;
    Lcuda_bin(Lcuda >= 1) = 1;
    fprintf('Number of unrecognized object pixels: \t%10s%10d\n','[CUDA]',sum( double(BIN(:))-Lcuda_bin(:))  );
    fprintf('...end\n')
% end
%% [deep] compare histogram consistency, object-by-object
if deep_check
    Ucu = unique(Lcuda(:));
    Uml = unique(Lml(:));
    % delete the background pixels from bins:
    Ucu(1) = [];
    Uml(1) = [];

    parfor ii = 1:numel(Ucu)
        Fcu     = Lcuda==Ucu(ii);
        [r,c]   = find(Fcu,1,'first');
        IDml    = Lml(r,c);
        Fml     = Lml==IDml;
        % compare the same object between Cuda and Matlab:
        isOne   = sum(abs( Fcu(:) - Fml(:) ));

        if isOne
            fprintf( '%5d, cu[ID=%10d,No=%4d], ml[ID=%4d,No=%4d]\n',ii,Ucu(ii),sum(Fcu(:)),IDml,sum(Fml(:)) );
        else
            fprintf('%d\n',ii)
        end
    end
end
% printed_this = [406,116,115,114,113,112,111,58,57,56,55,54,53,348,347,290,289,288,232,231,230,229,174,173,172,171,170,52,346,228,405,110,459,287,169,51,227,404,109,458,345,286,507,168,50,551,457,344,285,506,226,167,592,403,108,49,456,343,550,225,166,591,402,107,48,284,505,455,342,549,165,629,590,401,106,47,283,504,224,341,548,628,589,400,454,282,223,164,105,46,547,503,627,588,453,340,222,163,399,104,45,546,281,502,626,452,339,162,587,398,103,44,545,280,501,221,625,338,161,586,397,102,43,451,544,279,220,624,500,585,396,42,450,337,543,219,160,101,278,499,623,584,41,449,336,542,159,395,100,277,498,218,622,40,448,335,158,583,394,99,541,276,497,217,621,39,334,582,447,540,275,496,157,620,393,98,333,216,581,38,446,539,156,619,392,97,274,495,215,37,332,155,618,580,445,538,494,391,96,36,331,273,214,617,579,444,537,493,154,390,95,35,330,272,213,616,443,153,578,34,329,536,492,615,389,94,442,271,212,152,33,614,577,93,328,535,270,491,211,388,32,441,151,613,576,327,534,269,490,210,387,92,31,440,150,612,575,326,268,209,386,91,439,533,489,149,574,30,267,208,611,438,325,532,488,148,573,385,90,29,266,207,610,437,324,487,147,384,89,28,531,206,609,572,436,323,265,146,383,88,27,530,486,205,608,571,435,264,145,382,87,26,322,529,485,204,607,570,25,434,321,263,144,569,381,86,528,484,203,606,433,262,143,568,380,85,24,320,527,483,202,605,432,261,142,567,379,84,23,319,201,604,526,482,378,83,22,431,318,260,200,141,603,566,525,481,82,21,430,317,259,140,602,565,377,199,20,429,524,480,139,564,376,81,316,258,198,601,523,479,138,563,375,80,19,428,315,257,197,600,137,562,79,18,427,314,522,478,196,599,374,256,561,17,426,313,521,477,195,136,598,373,78,255,560,16,425,194,135,597,372,77,312,520,254,476,559,15,424,134,596,311,519,475,193,558,371,76,14,423,253,133,595,310,192,370,75,518,474,132,557,13,422,252,191,594,309,517,473,131,556,369,74,12,421,593,308,251,190,555,516,472,130,368,73,11,420,250,189,663,554,307,515,471,129,367,72,10,419,249,188,662,553,306,470,128,366,71,9,418,514,661,305,248,187,552,127,365,70,8,417,513,469,660,247,186,694,7,416,304,512,468,126,659,364,69,246,185,693,303,511,467,125,658,68,6,415,245,184,692,363,302,510,657,466,183,124,67,5,414,244,656,691,362,301,509,465,182,123,66,413,243,655,690,4,508,361,300,464,181,122,654,65,412,242,689,3,723,180,360,299,463,121,653,688,64,2,411,722,241,359,298,462,179,120,652,687,410,721,63,1,240,178,358,297,461,119,651,686,409,720,177,62,749,239,650,357,296,719,460,685,408,176,118,649,61,748,238,684,356,295,718,773,175,117,648,407,60,747,237,683,355,294,717,772,795,647,834,815,59,746,682,354,293,716,236,771,794,646,833,814,681,353,851,745,715,235,770,793,645,832,292,813,680,352,850,744,714,769,792,644,291,234,679,831,812,351,849,743,713,233,768,791,643,830,867,811,678,350,848,742,712,790,642,829,866,881,767,677,349,711,789,810,641,847,741,828,766,676,865,710,880,788,809,894,846,740,827,765,640,675,709,893,845,864,879,787,808,639,674,739,826,764,892,844,863,708,878,786,807,638,825,763,673,843,738,707,877,785,637,891,824,862,806,672,842,737,706,762,636,890,861,876,784,671,823,705,761,805,635,841,736,875,783,889,822,860,704,804,634,670,840,735,874,760,782,888,859,703,633,669,839,821,873,759,781,803,734,668,887,820,858,702,802,632,838,733,872,758,780,631,667,886,819,857,701,837,871,757,779,801,732,630,666,818,700,885,836,856,870,756,778,800,665,731,817,699,906,835,855,869,755,777,884,799,905,664,816,917,730,698,754,776,868,936,883,854,798,904,729,927,697,753,775,882,916,853,945,903,935,774,797,926,696,752,954,915,728,852,944,902,934,796,925,963,695,901,953,914,727,972,943,751,933,962,981,900,924,990,942,750,952,913,726,971,980,932,923,941,961,899,951,912,725,970,989,999,931,922,940,960,979,898,911,724,969,988,998,930,950,978,897,910,921,968,939,959,929,949,1008,987,997,967,938,977,896,948,909,1007,920,986,996,958,928,895,937,947,908,1006,919,966,976,1017,985,995,957,1026,946,965,1034,975,1005,918,984,907,994,956,1025,1016,1033,964,974,1004,983,993,955,1024,1015,1003,1032,973,982,992,1014,1023,1002,1031,991,1013,1030,1022,1012,1001,1021,1000,1029,1011,1020,1028,1010,1027,1019,1009,1018];
% sum( diff( sort( printed_this ) ) -1 ) % =0 !!!
%% plot
% figure(1),subplot(131),gpcolor(DIFFER),title('(CUDA - MatLab) CCL differences')
% figure(1),subplot(132),gpcolor(Lml),title('MatLab')
% figure(1),subplot(133),gpcolor(Lcuda),title('CUDA')
%% speedup
% fprintf('\n');
% fprintf('%8s%10s: %7.1f [ms]\n', 'Speed','[MatLab]', T(1)*1000 );
% fprintf('%8s%10s: %7.1f [ms]\n', 'Speed','[CUDA]', T(2)*1000 );
% fprintf('%8s%10s: %7.1f [--]\n', 'Speedup','[ml/cu]',T(1) / T(2) );
