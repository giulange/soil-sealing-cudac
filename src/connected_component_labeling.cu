/*
	Object:		Raster-scan and label-equivalence-based algorithm.
	Authors:	Giuliano Langella & Massimo Nicolazzo
	email:		gyuliano@libero.it


-----------
DESCRIPTION:
-----------

 I: "urban"		--> [0,0] shifted
 O: "lab_mat"	--> [1,1] shifted

	The "forward scan mask" for eight connected connectivity is the following:
		nw		nn		ne
		ww		cc		xx
		xx		xx		xx
	assuming that:
		> cc is the background(=0)/foreground(=1) pixel at (r,c),
		> nw, nn, ne, ww are the north-west, north, north-east and west pixels in the eight connected connectivity,
		> xx are skipped pixels.
	Therefore the mask has 4 active pixels with(out) object pixels (that is foreground pixels).

	In lab_mat pixels are ordered by blocks (first 1024 pixels by block[0], and so forth).

*/

//	INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>        	/* errno */
#include <string.h>       	/* strerror */
#include <math.h>			// ceil
#include <time.h>			// CLOCKS_PER_SEC

// GIS
#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis.h"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

//	-indexes
#define durban(cc,rr,bdx)	urban[		(cc)	+	(rr)	*(bdx)	] // I: scan value at current [r,c]
#define nw_pol(cc,rr,bdx)	lab_mat_sh[	(cc-1)	+	(rr-1)	*(bdx)	] // O: scan value at North-West
#define nn_pol(cc,rr,bdx)	lab_mat_sh[	(cc+0)	+	(rr-1)	*(bdx)	] // O: scan value at North
#define ne_pol(cc,rr,bdx)	lab_mat_sh[	(cc+1)	+	(rr-1)	*(bdx)	] // O: scan value at North-East
#define ww_pol(cc,rr,bdx)	lab_mat_sh[	(cc-1)	+	(rr+0)	*(bdx)	] // O: scan value at West
#define ee_pol(cc,rr,bdx)	lab_mat_sh[	(cc+1)	+	(rr+0)	*(bdx)	] // O: scan value at West
#define sw_pol(cc,rr,bdx)	lab_mat_sh[	(cc-1)	+	(rr+1)	*(bdx)	] // O: scan value at South-West
#define ss_pol(cc,rr,bdx)	lab_mat_sh[	(cc+0)	+	(rr+1)	*(bdx)	] // O: scan value at South-West
#define se_pol(cc,rr,bdx)	lab_mat_sh[	(cc+1)	+	(rr+1)	*(bdx)	] // O: scan value at South-West
#define cc_pol(cc,rr,bdx)	lab_mat_sh[	(cc+0)	+	(rr+0)	*(bdx)	] // O: scan value at current [r,c] which is shifted by [1,1] in O

__device__ unsigned int fBDX(unsigned int WIDTH){
	return (blockIdx.x < gridDim.x-1) ? blockDim.x : WIDTH  - (gridDim.x-1)*blockDim.x;// DO NOT MODIFY !!!!
}
__device__ unsigned int fBDY(unsigned int HEIGHT){ // NOTE: I'm assuming that blockDim.x = blockDim.y
	return (blockIdx.y < gridDim.y-1) ? blockDim.x : HEIGHT - (gridDim.y-1)*blockDim.x;// DO NOT MODIFY !!!!
}
__device__ unsigned int fBDY_cross(unsigned int HEIGHT){ // NOTE: I'm assuming that blockDim.x = blockDim.y
	return (blockIdx.y < gridDim.y-1) ? blockDim.x : HEIGHT - (gridDim.y-1)*blockDim.x;// DO NOT MODIFY !!!!
}


/*	+++++DEFINEs+++++	*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*	+++++DEFINEs+++++	*/

// GLOBAL VARIABLES
#define						Vo							1		// object value
#define						Vb							0		// object value
const bool					relabel						= true; // decide if relabel objects from 1 to N
static const unsigned int 	threads 					= 512;	//[reduce6] No of threads working in single block
static const unsigned int 	blocks 						= 64;	//[reduce6] No of blocks working in grid (this gives also the size of output Perimeter, to be summed outside CUDA)
const char 					*BASE_PATH					= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing";
char						buffer[255];
// I/-
// create on-the-fly in MatLab
const char 		*FIL_BIN	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/created-on-the-fly_BIN.tif";
const char 		*FIL_ROI	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/created-on-the-fly_ROI.tif";
// size=[300, 204]
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/fragmentation/data/ROI.tif";
//const char 		*FIL_BIN 		= "/home/giuliano/git/cuda/fragmentation/data/BIN.tif";
// size=[9152, 9002] **THIS GRIDS ARE BAD TO TEST CCL**
//const char 		*FIL_ROI 	= "/home/giuliano/git/cuda/fragmentation/data/lodi1954_roi.tif";
//const char 		*FIL_BIN 	= "/home/giuliano/git/cuda/fragmentation/data/lodi1954.tif";
// size=[9152, 9002]
//const char 		*FIL_ROI        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954_roi.tif";
//const char		*FIL_BIN        = "/media/DATI/wg-pedology/db-backup/LIFE+/50_Lodi/urban/lodi1954.tif";
// size=[15958, 15366]
//const char 		*FIL_ROI		= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped_roi.tif";
//const char 		*FIL_BIN		= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped.tif";
// size=[15001, 12001]
//const char		*FIL_ROI		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2_roi.tif";
//const char		*FIL_BIN		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/ispra/imp_mosaic_char_2006_cropped2.tif";
// size=[8000, 8000]
//const char 		*FIL_ROI		= "/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels_roi.tif";
//const char 		*FIL_BIN		= "/home/giuliano/git/cuda/perimeter/data/imp_mosaic_char_2006_cropped_64kpixels.tif";


// -/O
const char		*Lcuda		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/CUDA-code.txt";
const char		*Lhist		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/cu_histogram.txt";
const char 		*FIL_LAB 	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/LAB-MAT-cuda.tif";
// kernel_names
const char 		*kern_1 	= "intra_tile_labeling";
const char 		*kern_2 	= "stitching_tiles";
const char 		*kern_3 	= "root_equivalence";
const char 		*kern_4 	= "intra_tile_re_label";
const char 		*kern_4_a 	= "count_labels";
const char 		*kern_4_b 	= "labels__1_to_N";
const char 		*kern_4_c 	= "intratile_relabel_1toN";
const char 		*kern_5 	= "del_duplicated_lines";
const char 		*kern_6 	= "reduce6_hist";

//---------------------------- FUNCTIONS PROTOTYPES
//		** I/O **
void read_urbmat(unsigned char *, unsigned int, unsigned int, const char *);
void write_labmat_tiled( unsigned int *, unsigned int *, unsigned int *, unsigned int *, unsigned int *,  unsigned int, unsigned int, const char *);
void write_labmat_matlab(unsigned int *,  unsigned int, unsigned int, unsigned int, unsigned int, const char *);
//		** kernels **
//	(1)
__global__ void intra_tile_labeling( const unsigned char *,unsigned int, unsigned int, unsigned int, unsigned int * );
//	(2)
__global__ void stitching_tiles( unsigned int *,const unsigned int, const unsigned int, const unsigned int );
//	(3)
__global__ void root_equivalence( unsigned int *,const unsigned int,const unsigned int );
//	(4)
__global__ void intra_tile_re_label(unsigned int,unsigned int *);
//---------------------------- FUNCTIONS PROTOTYPES

// is Power two?
bool isPow2(unsigned int x){ return ((x&(x-1))==0); }

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

void
read_urbmat(unsigned char *urban, unsigned int nrows, unsigned int ncols, const char *filename)
{
	/*
	 * 	This function reads the Image and stores it in RAM with a 1-pixel-width zero-padding
	 */
	unsigned int rr,cc;
	FILE *fid ;
	int a;
	fid = fopen(filename,"rt");
	if (fid == NULL) { printf("Error opening file:\n\t%s\n",filename); exit(1); }
	for(rr=0;rr<nrows;rr++) for(cc=0;cc<ncols;cc++) urban[cc+rr*ncols] = 0;
	for(rr=1;rr<nrows-1;rr++){
		for(cc=1;cc<ncols-1;cc++){
			int out=fscanf(fid, "%d",&a);
			urban[cc+rr*ncols]=(unsigned char)a;
			//printf("%d ",a);
		}
		//printf("\n");
	}
	fclose(fid);
}
void
write_labmat_tiled(	unsigned int *lab_mat,
							unsigned int bdy, unsigned int bdx,
							unsigned int ntilesY, unsigned int ntilesX,
							unsigned int HEIGHT, unsigned int WIDTH,
							const char *filename)
{
	unsigned int rr,cc,bix,biy,tiDiX,tiDiY;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }
	int offset;

	for(biy=0;biy<ntilesY;biy++)
	{
		for(rr=0;rr<bdy;rr++)
		{
			for(bix=0;bix<ntilesX;bix++)
			{
				for(cc=0;cc<bdx;cc++)
				{
					/*if( !(((cc==nc-1) && ((ntilesX*biy+bix+1)%ntilesX)==0))	&&	// do not print last column
						!(((rr==nr-1) && (biy==ntilesY-1))) 						// do not print last row
					)*/
					if( bix*bdx+cc<WIDTH && biy*bdy+rr<HEIGHT)
					{
						/*
						 * 	This is the offset used in intra_tile_labeling:
						 * 		(WIDTH_e*bdy*biy) + (bdx*fBDY(HEIGHT_e)*bix) + (fBDX(WIDTH_e)*r+c)
						 * 	and this must be the offset for properly printing in file in HDD:
						 *
						 */
						tiDiY = bdy;
						if(biy==ntilesY-1) tiDiY = HEIGHT - (ntilesY-1)*bdy; // HEIGHT - (gridDim.y-1)*blockDim.y;
						tiDiX = bdx;
						if(bix==ntilesX-1) tiDiX = WIDTH  - (ntilesX-1)*bdx; // WIDTH  - (gridDim.x-1)*blockDim.x;
						offset = (WIDTH*biy*bdy) + (tiDiY*bdx*bix) + (tiDiX*rr+cc);
						fprintf(fid, "%6d ",lab_mat[offset]);
						//printf(		 "%d ",lab_mat[offset]);
					}
				}
				fprintf(fid,"\t\t");
				//printf(		"\n");
			}
			fprintf(fid,"\n");
			//printf(		"\n");
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}
void
write_labmat_tiled_without_duplicated_LINES(
							unsigned int *lab_mat,
							unsigned int bdy, unsigned int bdx,
							unsigned int ntilesY, unsigned int ntilesX,
							unsigned int HEIGHT, unsigned int WIDTH,
							const char *filename)
{
	/*
	 * 	THIS FUNCTION IS INCOMPLETE!!!
	 */

	unsigned int rr,cc,bix,biy,tiDiX,tiDiY;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }
	int offset;

	for(biy=0;biy<ntilesY;biy++)
	{
		for(rr=0;rr<bdy;rr++)
		{
			for(bix=0;bix<ntilesX;bix++)
			{
				for(cc=0;cc<bdx;cc++)
				{
					// do not print outside X/Y boundaries:
					if( bix*bdx+cc<WIDTH && biy*bdy+rr<HEIGHT )
					{
						/*
						 * 	This is the offset used in intra_tile_labeling:
						 * 		(WIDTH_e*bdy*biy) + (bdx*fBDY(HEIGHT_e)*bix) + (fBDX(WIDTH_e)*r+c)
						 * 	and this must be the offset for properly printing in file in HDD:
						 *
						 */
						tiDiY = bdy;
						if(biy==ntilesY-1) tiDiY = HEIGHT - (ntilesY-1)*bdy; // HEIGHT - (gridDim.y-1)*blockDim.y;
						tiDiX = bdx;
						if(bix==ntilesX-1) tiDiX = WIDTH  - (ntilesX-1)*bdx; // WIDTH  - (gridDim.x-1)*blockDim.x;
						offset = (WIDTH*biy*bdy) + (tiDiY*bdx*bix) + (tiDiX*rr+cc);
						fprintf(fid, "%6d ",lab_mat[offset]);
						//printf(		 "%d ",lab_mat[offset]);
					}
				}
				fprintf(fid,"\t\t");
				//printf(		"\n");
			}
			fprintf(fid,"\n");
			//printf(		"\n");
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}
void
write_labmat_full(unsigned int *lab_mat, unsigned int HEIGHT, unsigned int WIDTH, const char *filename)
{
	unsigned int rr,cc;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }

	for(rr=0;rr<HEIGHT;rr++)
	{
//		if(rr>0 && rr%32==0) fprintf(fid,"\n");
		for(cc=0;cc<WIDTH;cc++)
		{
//			if(cc>0 && cc%32==0) fprintf(fid,"\t");
			fprintf(fid, "%6d ",lab_mat[WIDTH*rr+cc]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}
void
write_labmat_matlab(unsigned int *lab_mat, unsigned int nr, unsigned int nc, unsigned int ntilesX, unsigned int ntilesY, const char *filename)
{
	unsigned int rr,cc,itX,itY;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }
	int offset;

	for(itY=0;itY<ntilesY;itY++)
	{
		for(rr=0;rr<nr;rr++)
		{
			for(itX=0;itX<ntilesX;itX++)
			{
				for(cc=0;cc<nc;cc++)
				{
					if( !(cc==nc-1)		&&	// do not print last column
						!(rr==nr-1) 	    // do not print last row
					)
					{
						offset = (ntilesX*itY+itX)*nc*nr+(nc*rr+cc);
						fprintf(fid, "%6d ",lab_mat[offset]);
						//printf(		 "%d ",lab_mat[offset]);
					}
				}
			}
			fprintf(fid,"\n");
			//printf(		"\n");
		}
	}
	fclose(fid);
}

int
delete_file( const char *file_name )
{
   int status = remove(file_name);

   if( status == 0 )
      printf("%s file deleted successfully.\n",file_name);
   else
   {
      printf("Unable to delete the file %s\n", file_name);
      perror("Error");
   }
   return 0;
}

__global__ void
intra_tile_labeling(const unsigned char *urban,unsigned int WIDTH,unsigned int HEIGHT,unsigned int WIDTH_e,unsigned int HEIGHT_e,unsigned int *lab_mat)
{
	/*
	 * 	*urban:		binary geospatial array;
	 * 	WIDTH:		number of columns of *urban;
	 * 	HEIGHT:		number of rows 	  of *urban;
	 * 	*lab_mat:	array of (intra-tile) labels, of size WIDTH_e*HEIGHT_e (see the main for size).
	 *
	 * 	NOTE: the use of "-1" in tix and tiy definition allows to account for duplicated adjacent LINES.
	 */

	// See this link when using more then one extern __shared__ array:
	// 		http://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable
	extern __shared__ unsigned int  lab_mat_sh[];

	__shared__ bool found;

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
//	unsigned int gdx		= gridDim.x;
//	unsigned int gdy		= gridDim.y;

	unsigned int tix		= (bdx-1)*bix + c;	// horizontal 	offset [the "-1" is necessary to duplicate adjacent LINES]
	unsigned int tiy		= (bdy-1)*biy + r;	// vertical 	offset [the "-1" is necessary to duplicate adjacent LINES]
	unsigned int itid		= tiy*WIDTH + tix;
	unsigned int otid		= bdx * r + c;

	unsigned int y_offset	= (WIDTH_e*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT_e)*bix);
	unsigned int blk_offset	= (fBDX(WIDTH_e)*r+c);
	unsigned int ttid		= y_offset + x_offset + blk_offset;

	if( tix<WIDTH && tiy<HEIGHT )
	{
		lab_mat_sh[otid] 	= 0;
		// if (r,c) is object pixel
		if  (urban[itid]==Vo)  lab_mat_sh[otid] = ttid;
		__syncthreads();

		found = true;
		while(found)
		{
			/* 		________________
			 * 		|	 |    |    |
			 *		| nw | nn | ne |
			 *		|____|____|____|
			 * 		|	 |    |    |
			 * 		| ww | cc | ee |	pixel position
			 *		|____|____|____|
			 * 		|	 |    |    |
			 * 		| sw | ss | se |
			 * 		|____|____|____|
			 */
			found = false;

			// NW:
			if(	c>0 && r>0 && nw_pol(c,r,bdx)!=0 && nw_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = nw_pol(c,r,bdx); found = true; }
			// NN:
			if( r>0 && nn_pol(c,r,bdx)!=0 && nn_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = nn_pol(c,r,bdx); found = true; }
			// NE:
			if( c<fBDX(WIDTH_e)-1 && r>0 && ne_pol(c,r,bdx)!=0 && ne_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = ne_pol(c,r,bdx); found = true; }
			// WW:
			if( c>0 && ww_pol(c,r,bdx)!=0 && ww_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = ww_pol(c,r,bdx); found = true; }
			// EE:
			if( c<fBDX(WIDTH_e)-1 && ee_pol(c,r,bdx)!=0 && ee_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = ee_pol(c,r,bdx); found = true; }
			// SW:
			if( c>0 && r<fBDY(HEIGHT_e)-1 && sw_pol(c,r,bdx)!=0 && sw_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = sw_pol(c,r,bdx); found = true; }
			// SS:
			if( r<fBDY(HEIGHT_e)-1 && ss_pol(c,r,bdx)!=0 && ss_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = ss_pol(c,r,bdx); found = true; }
			// SE:
			if( c<fBDX(WIDTH_e)-1 && r<fBDY(HEIGHT_e)-1 && se_pol(c,r,bdx)!=0 && se_pol(c,r,bdx)<cc_pol(c,r,bdx))
				{ cc_pol(c,r,bdx) = se_pol(c,r,bdx); found = true; }

			__syncthreads();
		}

		/*
		 * 	I write using ttid, therefore I linearize the array with respect to blocks.
		 */
		lab_mat[ttid] = lab_mat_sh[otid];
		__syncthreads();
	}
}
template <unsigned int NTHREADSX>
__global__ void
stitching_tiles(	unsigned int *lab_mat,
									const unsigned int bdy,
									const unsigned int WIDTH_e,
									const unsigned int HEIGHT_e){
	/**
	 * 	This kernel stitches adjacent tiles working on borders.
	 *
	 * 	NOTE:
	 * 		> xx_yy is the tile xx and border yy (e.g. nn_ss is tile at north and border at south).
	 */

	//unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int tix 		= bdx*bix + c;
	unsigned int tiy 		= bdy*biy + c; // ATTENTION: here I use c instead of r, because the block is defined on X, and Y=1 !!!

	// SIDES:	(I use fBDY_cross because blockDim.y=1, while I need the real size of tile on Y, which I set equal to bdx)
	int c_nn_tid		=	c 										+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int nn_tid			= 	c +	fBDX(WIDTH_e)*(bdy-1)				+	// (3) within-tile offset
							(bdx*bdy*bix)							+	// (2) X offset				// fBDY_cross(HEIGHT_e)
							(WIDTH_e*bdy*(biy-1))					;	// (1) Y offset

	int c_ww_tid		=	c*fBDX(WIDTH_e)							+	// (3) within-tile offset	// fBDX(WIDTH_e)
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int ww_tid			= 	(c+1)*bdx-1								+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*(bix-1))		+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	// SHARED: "tile_border" ==> cc_nn is border North of Centre tile
	__shared__ unsigned int cc_nn[NTHREADSX];
	__shared__ unsigned int nn_ss[NTHREADSX];
	__shared__ unsigned int cc_ww[NTHREADSX];
	__shared__ unsigned int ww_ee[NTHREADSX];
	__shared__ unsigned int __old[NTHREADSX];
	__shared__ unsigned int _min_[NTHREADSX];
	__shared__ unsigned int _max_[NTHREADSX];

	//unequal[c] = false;
	// ...::NORTH::...
	if( biy>0 && tix<WIDTH_e ){
		//recursion ( lab_mat, c_nn_tid, nn_tid );
		/*
		 * 		(1) **list** { cc_nn(i), nn_ss(i) }
		 */
		cc_nn[ c ]		= lab_mat[ c_nn_tid ]; 	__syncthreads();			// __max
		nn_ss[ c ] 		= lab_mat[ nn_tid ];	__syncthreads();			// __min
/*		if( (cc_nn[c]==0 & nn_ss[c]!=0) | (cc_nn[c]!=0 & nn_ss[c]==0) )
			unequal[c] = true;
*/
		/*
		 * 		(2) **recursion applying split-rules**
		 */
		__old[ c ] = atomicMin( &lab_mat[ cc_nn[c] ], nn_ss[ c ] ); // write the current min val where the index cc_nn[c] is in lab_mat.
		//__syncthreads();
		while( __old[ c ] != nn_ss[c] )
		{
			_min_[ c ] 	= ( (nn_ss[c]) < (__old[c]) )? nn_ss[c] : __old[c];
			_max_[ c ] 	= ( (nn_ss[c]) > (__old[c]) )? nn_ss[c] : __old[c];
			__old[ c ] 	= atomicMin( &lab_mat[ _max_[c] ], _min_[ c ] );
			nn_ss[ c ] 	= _min_[ c ];
		}
		__syncthreads();
	}

	// ...::WEST::...
	if( bix>0 && tiy<HEIGHT_e){
		//recursion ( lab_mat, c_ww_tid, ww_tid );
		/*
		 * 		(1) **list** { cc_nn(i), nn_ss(i) }
		 */
		cc_ww[ c ]		= lab_mat[ c_ww_tid ];	__syncthreads();
		ww_ee[ c ] 		= lab_mat[ ww_tid ];	__syncthreads();
		//__syncthreads();
/*		if( (cc_ww[c]==0 & ww_ee[c]!=0) | (cc_ww[c]!=0 & ww_ee[c]==0) )
			unequal[c] = true;
*/
		/*
		 * 		(2) **recursion applying split-rules**
		 */
		__old[ c ] = atomicMin( &lab_mat[ cc_ww[c] ], ww_ee[ c ] );
		//__syncthreads();
		while( __old[ c ] != ww_ee[c] )
		{
			_min_[ c ] 	= ( (ww_ee[c]) < (__old[c]) )? (ww_ee[c]) : (__old[c]);
			_max_[ c ] 	= ( (ww_ee[c]) > (__old[c]) )? (ww_ee[c]) : (__old[c]);
			__old[ c ] 	= atomicMin( &lab_mat[ _max_[c] ], _min_[ c ] );
			ww_ee[ c ] 	= _min_[ c ];//lab_mat[ _max_[c] ];
		}
		__syncthreads();
	}
}
template <unsigned int NTHREADSX>
__global__ void
root_equivalence(	unsigned int *lab_mat,
									const unsigned int bdy,
									const unsigned int WIDTH_e,
									const unsigned int HEIGHT_e		){
	/**
	 * 	This kernel finds the root ID for any key-pixel.
	 * 	A key-pixel is a pixel in a block having lab_mat[tid]=tid.
	 */

	//unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
//	unsigned int bdy		= tiledimY; //blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int tix 		= bdx*bix + c;
	unsigned int tiy 		= bdy*biy + c; // ATTENTION: here I use c instead of r, because the block is defined on X, and Y=1 !!!

	// SIDES:
	int c_nn_tid		=	c 										+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int nn_tid			= 	c +	fBDX(WIDTH_e)*(bdy-1)				+	// (3) within-tile offset
							(bdx*bdy*bix)							+	// (2) X offset
							(WIDTH_e*bdy*(biy-1))					;	// (1) Y offset

	int c_ww_tid		=	c*fBDX(WIDTH_e)							+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int ww_tid			= 	(c+1)*bdx-1								+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*(bix-1))		+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int c_ss_tid		=	c + fBDX(WIDTH_e)*(fBDY_cross(HEIGHT_e)-1)+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int ss_tid			= 	c										+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*(biy+1))					;	// (1) Y offset

	int c_ee_tid		=	(c+1)*bdx-1								+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*bix)			+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	int ee_tid			= 	c*fBDX(WIDTH_e)							+	// (3) within-tile offset
							(bdx*fBDY_cross(HEIGHT_e)*(bix+1))		+	// (2) X offset
							(WIDTH_e*bdy*biy)						;	// (1) Y offset

	// SHARED:
	__shared__ unsigned int cc_nn[NTHREADSX];
	__shared__ unsigned int nn_ss[NTHREADSX];
	__shared__ unsigned int cc_ww[NTHREADSX];
	__shared__ unsigned int ww_ee[NTHREADSX];
	__shared__ unsigned int cc_ss[NTHREADSX];
	__shared__ unsigned int ss_nn[NTHREADSX];
	__shared__ unsigned int cc_ee[NTHREADSX];
	__shared__ unsigned int ee_ww[NTHREADSX];

	// ...::NORTH::...
	if( biy>0 && tix<WIDTH_e ){
		/*
		 * 		(1) **list** { cc_nn(i), nn_ss(i) }
		 */
		cc_nn[ c ]		= lab_mat[ c_nn_tid ];
		nn_ss[ c ] 		= lab_mat[ nn_tid ]; // --> DELETE, because nn_cc = cc_nn after!!
		__syncthreads();

		/*
		 * 		(2) **recursion finding root equivalence** nn_ss(i) = lab_mat[ nn_ss(i) ]

					lab_mat[ nn_ss(i) ]  ---> ID(t)   ------|
															| 	with recursion on t
					lab_mat[ ID(t) ]     ---> ID(t+1) <-----|
		 */
		nn_ss[ c ]		= cc_nn[c];
		while( lab_mat[ nn_ss[c] ] != nn_ss[c] )
		{
			{
				nn_ss[ c ] 	= lab_mat[ nn_ss[c] ];
				__threadfence_system();//__syncthreads();
			}
		}
		atomicMin( &lab_mat[ cc_nn[ c ] ], nn_ss[c] );
		atomicMin( &lab_mat[ c_nn_tid ],   nn_ss[c] );
		//lab_mat[ c_nn_tid ] = lab_mat[ nn_ss[c] ];
		__syncthreads();
	}

	// ...::WEST::...
	if( bix>0 && tiy<HEIGHT_e){
		/*
		 * 		(1) **list** { cc_nn(i), nn_ss(i) }
		 */
		cc_ww[ c ]		= lab_mat[ c_ww_tid ];
		ww_ee[ c ] 		= lab_mat[ ww_tid ];
		__syncthreads();

		/*
		 * 		(2) **recursion finding root equivalence** nn_ss(i) = lab_mat[ nn_ss(i) ]

					lab_mat[ nn_ss(i) ]  ---> ID(t)   ------|
															| 	with recursion on t
					lab_mat[ ID(t) ]     ---> ID(t+1) <-----|
		 */
		ww_ee[ c ]		= cc_ww[c];
		while( lab_mat[ ww_ee[c] ] != ww_ee[c] )
		{
			ww_ee[ c ] 	= lab_mat[ ww_ee[c] ];
			__threadfence_system();//__syncthreads();
		}
		atomicMin( &lab_mat[ cc_ww[ c ] ], ww_ee[c] );
		atomicMin( &lab_mat[ c_ww_tid ],   ww_ee[c] );
		//lab_mat[ c_ww_tid ] = lab_mat[ ww_ee[c] ];
		__syncthreads();//__threadfence_system();
	}

	// ...::SOUTH::...
	if( biy<gdy-1 && tix<WIDTH_e ){
		/*
		 * 		(1) **list** { cc_nn(i), nn_ss(i) }
		 */
		cc_ss[ c ]		= lab_mat[ c_ss_tid ];
		ss_nn[ c ] 		= lab_mat[ ss_tid ];
		__syncthreads();

		/*
		 * 		(2) **recursion finding root equivalence** nn_ss(i) = lab_mat[ nn_ss(i) ]

					lab_mat[ nn_ss(i) ]  ---> ID(t)   ------|
															| 	with recursion on t
					lab_mat[ ID(t) ]     ---> ID(t+1) <-----|
		 */
		ss_nn[ c ]		= cc_ss[c];
		while( lab_mat[ ss_nn[c] ] != ss_nn[c] )
		{
			ss_nn[ c ] 	= lab_mat[ ss_nn[c] ];
			__threadfence_system();//__syncthreads();
		}
		atomicMin( &lab_mat[ cc_ss[ c ] ], ss_nn[c] );
		atomicMin( &lab_mat[ c_ss_tid ],   ss_nn[c] );
		__syncthreads();
	}

	// ...::EAST::...
	if( bix<gdx-1 && tiy<HEIGHT_e ){
		/*
		 * 		(1) **list** { cc_nn(i), nn_ss(i) }
		 */
		cc_ee[ c ]		= lab_mat[ c_ee_tid ];
		ee_ww[ c ] 		= lab_mat[ ee_tid ];
		__syncthreads();

		/*
		 * 		(2) **recursion finding root equivalence** nn_ss(i) = lab_mat[ nn_ss(i) ]

					lab_mat[ nn_ss(i) ]  ---> ID(t)   ------|
															| 	with recursion on t
					lab_mat[ ID(t) ]     ---> ID(t+1) <-----|
		 */
		ee_ww[ c ]		= cc_ee[c];
		while( lab_mat[ ee_ww[c] ] != ee_ww[c] )
		{
			ee_ww[ c ] 	= lab_mat[ ee_ww[c] ];
			__threadfence_system();//__syncthreads();
		}
		atomicMin( &lab_mat[ cc_ee[ c ] ], ee_ww[c] );
		atomicMin( &lab_mat[ c_ee_tid ],   ee_ww[c] );
		__syncthreads();//__threadfence_system();
	}
}
__global__ void
intra_tile_re_label(unsigned int WIDTH_e, unsigned int HEIGHT_e, unsigned int *lab_mat){
	/**
	 * 	This kernel assign to each pixel the value of the root ID written where
	 * 	lab_mat[tid] is equal to tid.
	 */


	// See this link when using more then one extern __shared__ array:
	// 		http://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable
	//extern __shared__ unsigned char urban_sh[];
//	extern __shared__ unsigned int  lab_mat_sh[];

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
//	unsigned int gdx		= gridDim.x;
//	unsigned int gdy		= gridDim.y;
	unsigned int tix		= bdx*bix + c;	// horizontal 	offset
	unsigned int tiy		= bdy*biy + r;	// vertical 	offset

//	unsigned int otid		= bdx * r + c;

	unsigned int y_offset	= (WIDTH_e*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT_e)*bix);
	unsigned int blk_offset	= (fBDX(WIDTH_e)*r+c);
	unsigned int ttid		= y_offset + x_offset + blk_offset;

	if( tix<WIDTH_e && tiy<HEIGHT_e )// iTile<gdx*gdy
	{
		if(lab_mat[ttid]!=Vb)  lab_mat[ttid]=lab_mat[lab_mat[ttid]];
	}
}
__global__ void
count_labels( 	unsigned int WIDTH, unsigned int HEIGHT,
				const unsigned int *lab_mat, unsigned int *bins ){
	/**
	 * 	This kernel counts the number of objects within each block (and stores it in bins).
	 * 		1.\ build ones array (simple write operation)
	 * 		2.\ count ones within ones array (reduction) --> only works with NTHREADSX=32 !!!!
	 */

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
//	unsigned int gdy		= gridDim.y;
	unsigned int big		= biy*gdx + bix;	// block index on grid!
	//unsigned int tid 		= bdx * r + c;
	unsigned int tid 		= fBDX(WIDTH) * r + c;
	unsigned int tix		= bdx*bix + c;		// horizontal 	offset
	unsigned int tiy		= bdy*biy + r;		// vertical 	offset

//	unsigned int otid		= bdx * r + c;

	unsigned int y_offset	= (WIDTH*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT)*bix);
	unsigned int blk_offset	= (fBDX(WIDTH)*r+c);
	unsigned int ttid		= y_offset + x_offset + blk_offset;

	extern __shared__ unsigned int sh_sum[];
	//	-initialisation:
	sh_sum[tid] = 0;									syncthreads();

	if( tix<WIDTH && tiy<HEIGHT )
	{
		// 1.\ simple write operation
		//	-ones:
		if(lab_mat[ttid]==ttid) sh_sum[tid] = 1;		syncthreads();
		//	-the lab_mat[ttid]==ttid condition is not valid for label "ZERO":
		if(ttid==0) sh_sum[tid] = 0;

		// 2.\ reduction
		//	-compute sum:
		if(tid<512)	sh_sum[tid] += sh_sum[tid + 512];	syncthreads();
		if(tid<256)	sh_sum[tid] += sh_sum[tid + 256];	syncthreads();
		if(tid<128)	sh_sum[tid] += sh_sum[tid + 128];	syncthreads();
		if(tid<64)	sh_sum[tid] += sh_sum[tid + 64];	syncthreads();
		if(tid<32)	sh_sum[tid] += sh_sum[tid + 32];	syncthreads();
		if(tid<16)	sh_sum[tid] += sh_sum[tid + 16];	syncthreads();
		if(tid<8)	sh_sum[tid] += sh_sum[tid + 8];		syncthreads();
		if(tid<4)	sh_sum[tid] += sh_sum[tid + 4];		syncthreads();
		if(tid<2)	sh_sum[tid] += sh_sum[tid + 2];		syncthreads();
		if(tid<1)	sh_sum[tid] += sh_sum[tid + 1];		syncthreads();
		//	-assign sum to its block
		if(tid==0)	bins[big] = sh_sum[tid];
	}
}
__global__ void
labels__1_to_N( unsigned int WIDTH, 	unsigned int HEIGHT,
				unsigned int *lab_mat, 	unsigned int *cumsum,
				unsigned int kmax_e, 	unsigned int bdx_end,
				unsigned int *ID_rand, 	unsigned int *ID_1toN){
	/**
	 * 	This kernel writes in lab_mat[tid]=tid an ID values such that all IDs are between [1,Nbins].
	 */
	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int big		= biy*gdx + bix;// block index on grid!
	unsigned int tix		= bdx*bix + c;	// horizontal 	offset
	unsigned int tiy		= bdy*biy + r;	// vertical 	offset
	//unsigned int tid		= bdx * r + c;
	unsigned int tid		= fBDX(WIDTH) * r + c;

	unsigned int y_offset	= (WIDTH*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT)*bix);
	unsigned int blk_offset	= (fBDX(WIDTH)*r+c);
	unsigned int ttid		= y_offset + x_offset + blk_offset;
	unsigned int k 			= cumsum[big];
	unsigned int kmax 		= 0;
	if(big>=gdx*gdy-1) kmax	= kmax_e;
	else 			   kmax = cumsum[big+1];

	extern __shared__ unsigned int sh_sum[];
	//	-initialisation:
	sh_sum[tid] = 0; syncthreads();

	unsigned int bdx_act	= 0;
	unsigned int bdy_act	= 0;

	if( tix<WIDTH && tiy<HEIGHT && kmax-k>0)
	{
		// 0.\ prepare
		//if(blockIdx.x >= gridDim.x-1) bdx		= bdx_end;
		bdx_act = fBDX(WIDTH); // actual block dim in X
		bdy_act = fBDY(HEIGHT);// actual block dim in Y (In fBDY I assume that blockDim.X = blockDim.Y)

		// 1.\ simple write operation
		//	-ones:
		if(lab_mat[ttid]==ttid){ sh_sum[tid] = 1; syncthreads(); }
		//	-the lab_mat[ttid]==ttid condition is not valid for label "ZERO":
		if(ttid==0){ sh_sum[tid]= 0; }// && lab_mat[ttid]==Vb

		// 2.\ write labels within [1,Nbins] in lab_mat
		unsigned int ii=0;
		if (tid==0){// thread=0 writes
			for(unsigned int row=0;row<bdy_act;row++){
				for(unsigned int col=0;col<bdx_act;col++){
					if (sh_sum[row*bdx_act+col]==1){
						ii = row*bdx_act+col;
						ID_rand[k] = lab_mat[ttid+ii];
						ID_1toN[k] = k+1;
						lab_mat[ttid+ii] = k+1;
						k+=1;
						//lab_mat[ttid+ii] = k;
					}
				}
			}
		}
	}
}
__global__ void
intratile_relabel_1toN_notgood(	unsigned int WIDTH_e, unsigned int HEIGHT_e,
						unsigned int *lab_mat, unsigned int *cumsum, unsigned int Nbins,
						const unsigned int *ID_rand, const unsigned int *ID_1toN ){
	/**
	 * 	This kernel assign to each pixel the value of the root ID written where
	 * 	lab_mat[tid] is equal to tid.
	 */

	extern __shared__ unsigned int  lab_mat_sh[];

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	// new
	//unsigned int gdx		= gridDim.x;
	//unsigned int gdy		= gridDim.y;
	//unsigned int big		= biy*gdx + bix;	// block index on grid!
	// new
	unsigned int tix		= bdx*bix + c;		// horizontal 	offset
	unsigned int tiy		= bdy*biy + r;		// vertical 	offset
	unsigned int otid		= bdx * r + c;

	unsigned int y_offset	= (WIDTH_e*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT_e)*bix);
	unsigned int blk_offset	= (fBDX(WIDTH_e)*r+c);
	unsigned int ttid		= y_offset + x_offset + blk_offset;

	if( tix<WIDTH_e && tiy<HEIGHT_e )// iTile<gdx*gdy
	{
		lab_mat_sh[otid] = lab_mat[ttid]; 		syncthreads();

		// find the upper and lower IDs limits for current block:
		int start = 0;//big==0? 0:cumsum[big-1]+0;
		int end   = Nbins;//big==gdx*gdy-1? Nbins : cumsum[big+1];
		// try to write a sequence of IDs starting from 1 to N:
		//for(unsigned int ii=0; ii<Nbins; ii++){
		for(unsigned int ii=start; ii<=end; ii++){
			if(lab_mat_sh[otid]==ID_rand[ii]){
				lab_mat_sh[otid] = ID_1toN[ii]; syncthreads();
			}
		}
		lab_mat[ttid] = lab_mat_sh[otid]; 		syncthreads();
	}
}
__global__ void
intratile_relabel_1toN(unsigned int WIDTH_e, unsigned int HEIGHT_e, unsigned int *lab_mat){
	/**
	 * 	This kernel assign to each pixel the value of the root ID written where
	 * 	lab_mat[tid] is equal to tid.
	 */


	// See this link when using more then one extern __shared__ array:
	// 		http://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable
	//extern __shared__ unsigned char urban_sh[];
//	extern __shared__ unsigned int  lab_mat_sh[];

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
//	unsigned int gdx		= gridDim.x;
//	unsigned int gdy		= gridDim.y;
	unsigned int tix		= bdx*bix + c;	// horizontal 	offset
	unsigned int tiy		= bdy*biy + r;	// vertical 	offset

//	unsigned int otid		= bdx * r + c;

	unsigned int y_offset	= (WIDTH_e*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT_e)*bix);
	unsigned int blk_offset	= (fBDX(WIDTH_e)*r+c);
	unsigned int ttid		= y_offset + x_offset + blk_offset;

	if( tix<WIDTH_e && tiy<HEIGHT_e )// iTile<gdx*gdy
	{
		if(lab_mat[ttid]!=Vb && lab_mat[lab_mat[ttid]]!=Vb)  lab_mat[ttid]=lab_mat[lab_mat[ttid]];
	}
}
__global__ void
del_duplicated_lines( 	const unsigned int *lab_mat_gpu,	unsigned int WIDTH_e,unsigned int HEIGHT_e,
							  unsigned int *lab_mat_gpu_f,	unsigned int WIDTH,	 unsigned int HEIGHT	){
	/**
	 * 	This kernel delete duplicated lines and write final image with pixels
	 * 	ordered as they are located in geographical domain.
	 */
	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;

	// global offset:
	unsigned int tix		= bdx*bix + c;				// horizontal 	offset
	unsigned int tiy		= bdy*biy + r;				// vertical 	offset
	// for lab_mat_gpu:
	unsigned int y_offset	= (WIDTH_e*bdy*biy);
	unsigned int x_offset	= (bdx*fBDY(HEIGHT_e)*bix);
	unsigned int blk_offset	= fBDX(WIDTH_e)*(r+1)+c;	// +1 to skip first row
	unsigned int ttid		= y_offset + x_offset + blk_offset;
	// for lab_mat_gpu_f:
	unsigned int tix_f		= (bdx-1)*bix + c;			// horizontal 	offset
	unsigned int tiy_f		= (bdy-1)*biy + r;			// vertical 	offset
	unsigned int ttid_f		= tiy_f*WIDTH + tix_f;

	if( tix<WIDTH_e && tiy<HEIGHT_e ){
		// write all pixels, except last row:
		if(r<bdy-1)	lab_mat_gpu_f[ttid_f] = lab_mat_gpu[ttid];
	}
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_hist___original(const T *g_idata, const unsigned char *ROI, T *g_ohist, unsigned int map_len, unsigned int Nbins)
{
	/*  x---bdx*mapel_per_thread----x
	 * 	 ___________________________ __...
	 * 	|___________________________|__...		*g_idata
	 *
	 * 	 ___ ___ ___ ___ ___ ___ ___ __...
	 * 	|___|___|___|___|___|___|___|__...		*g_idata
	 * 	x---x
	 * 	 bdx
	 * 	|   \
	 * 	|    \____________________
	 * 	|     					  |				 -zoom in g_idata to highlight how sdata works.
	 * 	x-----------bdx-----------x				 -bdx=threads
	 * 	 _ _ _ _ _ _ _ _ _ _ _ _ _ _...
	 * 	|_|_|_|_|_|_|_|_|_|_|_|_|_|_...			sdata   _
	 * 	 						  			     -each |_| is an element of sdata in which mapel_per_thread pixels are summed up by one tid.
	 * 	 ____________  x
	 * 	|            | |
	 *  | *sdata     | Nbins					 -the mapel_per_thread pixels are summed up by tid and written using this offset:
	 *  |            | |							offset = j*bdx + tid; where j=1,...,Nbins.
	 *  |____________| x						 -each tid is in charge of its column number in sdata.
	 *											 -each row in sdata represents a bin "j" within Nbins.
	 *  x--threads---x
	 */
    T *sdata = SharedMemory<T>();// size = bdx * Nbins
    // sdata_j
	__shared__ unsigned int sdata_j[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid 		= threadIdx.x;
    unsigned int bix 		= blockIdx.x;
    unsigned int bdx 		= blockDim.x;
    unsigned int gdx 		= gridDim.x;
    unsigned int i 			= bix*blockSize*2 + tid;
    unsigned int gridSize 	= blockSize*2*gdx;
    unsigned int j			= 0;
    unsigned int offset 	= 0;

    T 			 *mySum;
    T			 locSum;

    //for(j=0;j<Nbins;j++) sdata[j*bdx+tid]=0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < map_len)
    {
    	mySum[ g_idata[i]*bdx ] += ROI[i];// here the j of the offset is given by the value in g_idata[i]
        // ensure we don't read out of bounds -- this is optimised away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < map_len) mySum[ g_idata[i+blockSize]*bdx ] += ROI[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

    /*
     * Now I have sdata that stores for each bin "j" in Nbins (y-axis) the sum of ID=j of object "j"
     * for each of the tid threads (x-axis) in current block.
     * The next step (i.e. the for loop on j) has to reduce each row of sdata so that each block
     * gives the histogram of its bdx*mapel_per_thread pixels.
     *
     * 	 ____________  x
	 * 	|            | |
	 *  | *sdata     | Nbins
	 *  |            | |
	 *  |____________| x
	 *
	 *  x--threads---x
	 */
    for(j=0;j<Nbins;j++){// start from j=1, to avoid the computation of background

    	offset 			= j*bdx+tid;
        // each thread puts the sum in shared memory into local memory
        locSum 			= mySum[j]; __syncthreads();
        sdata_j[tid] 	= locSum;//sdata[offset];
        __syncthreads();

		// do reduction in shared memory
		if (blockSize >= 512) if (tid < 256) sdata_j[tid] = locSum = locSum + sdata_j[tid + 256]; __syncthreads();
		if (blockSize >= 256) if (tid < 128) sdata_j[tid] = locSum = locSum + sdata_j[tid + 128]; __syncthreads();
		if (blockSize >= 128) if (tid <  64) sdata_j[tid] = locSum = locSum + sdata_j[tid +  64]; __syncthreads();
		if (tid < 32)
		{
			// now that we are using warp-synchronous programming (below)
			// we need to declare our shared memory volatile so that the compiler
			// doesn't reorder stores to it and induce incorrect behaviour.
			volatile T *smem = sdata_j;

			if (blockSize >=  64) smem[tid] = locSum = locSum + smem[tid + 32];
			if (blockSize >=  32) smem[tid] = locSum = locSum + smem[tid + 16];
			if (blockSize >=  16) smem[tid] = locSum = locSum + smem[tid +  8];
			if (blockSize >=   8) smem[tid] = locSum = locSum + smem[tid +  4];
			if (blockSize >=   4) smem[tid] = locSum = locSum + smem[tid +  2];
			if (blockSize >=   2) smem[tid] = locSum = locSum + smem[tid +  1];
		}
	    // write result for this block to global memory
		//if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	    if (tid == 0) atomicAdd( &g_ohist[ j ], sdata_j[0] );//cannot understand why NVidia does not put smem instead of sdata!!!
    }
}

template <class T>
__global__ void
reduce6_hist(const T *g_idata, const unsigned char *ROI, T *g_ohist, unsigned int map_len, unsigned int Nbins)
{
	/*  x---bdx*mapel_per_thread----x
	 * 	 ___________________________ __...
	 * 	|___________________________|__...		*g_idata
	 *
	 * 	 ___ ___ ___ ___ ___ ___ ___ __...
	 * 	|___|___|___|___|___|___|___|__...		*g_idata
	 * 	x---x
	 * 	 bdx
	 * 	|   \
	 * 	|    \____________________
	 * 	|     					  |				 -zoom in g_idata to highlight how sdata works.
	 * 	x-----------bdx-----------x				 -bdx=threads
	 * 	 _ _ _ _ _ _ _ _ _ _ _ _ _ _...
	 * 	|_|_|_|_|_|_|_|_|_|_|_|_|_|_...			sdata   _
	 * 	 						  			     -each |_| is an element of sdata in which mapel_per_thread pixels are summed up by one tid.
	 * 	 ____________  x
	 * 	|            | |
	 *  | *sdata     | Nbins					 -the mapel_per_thread pixels are summed up by tid and written using this offset:
	 *  |            | |							offset = j*bdx + tid; where j=1,...,Nbins.
	 *  |____________| x						 -each tid is in charge of its column number in sdata.
	 *											 -each row in sdata represents a bin "j" within Nbins.
	 *  x--threads---x
	 */
    //T *sdata = SharedMemory<T>();// size = bdx * Nbins
    // sdata_j
	extern __shared__ unsigned int sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid 		= threadIdx.x;
    unsigned int bix 		= blockIdx.x;
    unsigned int bdx 		= blockDim.x;
    unsigned int gdx 		= gridDim.x;
    unsigned int i 			= bix*bdx + tid;
    unsigned int gridSize 	= bdx*gdx;
    unsigned int j			= 0;

    for(j=0;j<=Nbins;j++) if(tid==0) sdata[j]=0; //sdata[j*bdx+tid]=0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < map_len)
    {
    	atomicAdd( &sdata[ g_idata[i] ], ROI[i] );//ROI[i] instead of 1
        i += gridSize;
    }
    __threadfence_system();
    __syncthreads();

    /*
     * Now I have sdata that stores for each bin "j" in Nbins (y-axis) the sum of ID=j of object "j"
     * for each of the tid threads (x-axis) in current block.
     * The next step (i.e. the for loop on j) has to reduce each row of sdata so that each block
     * gives the histogram of its bdx*mapel_per_thread pixels.
     *
     * 	 ____________  x
	 * 	|            | |
	 *  | *sdata     | Nbins
	 *  |            | |
	 *  |____________| x
	 *
	 *  x--threads---x
	 */
    for(j=0;j<=Nbins;j++){// start from j=1, to avoid the computation of background
	    // write result for this block to global memory
		//if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	    if (tid == 0) atomicAdd( &g_ohist[j], sdata[j] );//cannot understand why NVidia does not put smem instead of sdata!!!
    }
}

/** I have two issues:
 * 		> ccl algorithm does not work for: (i) large images[false], (ii) too much compacted objects[?], (iii) BIN with all ones[true], (iv) small images[?]
 * 		> histogram doesn't work at all
 */
int main(int argc, char **argv){

	/**
	 * 		D E L E T E   F I L E S
	 * 		rationale : if execution interrupts before writing the new files, I would consider wrong information in MatLan during the check!
	 */
	delete_file( FIL_LAB );
	delete_file( Lhist );


	// Parameter:
	const unsigned int 	NTHREADSX 			= 32;

	// DECLARATIONS:
	unsigned int		ii,Nbins;
	bool 				printme				= false;
	unsigned int  		*lab_mat_cpu, *lab_mat_gpu, *lab_mat_cpu_f, *lab_mat_gpu_f, *bins_cpu,/* *ones_gpu,*/ *bins_gpu, *cumsum, *ID_rand_gpu, *ID_1toN_gpu;
	unsigned char 		*urban_gpu,*dev_ROI;
	unsigned int 		*h_histogram, *d_histogram;// it's size is equal to the number of blocks within grid!
	metadata 			MDbin,MDuint,MDroi;
	unsigned int		gpuDev				= 0;
	unsigned int 		sqrt_nmax_threads 	= 0;
	unsigned int 		num_blocks_per_SM, mapel_per_thread;
	unsigned int		map_len;
	// it counts the number of kernels that must print their LAB-MAT:
	unsigned int 		count_print			= 0;
	unsigned int 		elapsed_time		= 0;
	// clocks:
	clock_t 			start_t, end_t;
	cudaDeviceProp		devProp;
	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	// query current GPU properties:
	CUDA_CHECK_RETURN( cudaSetDevice(gpuDev) );
	cudaGetDeviceProperties(&devProp, gpuDev);
	sqrt_nmax_threads 		= NTHREADSX;//floor(sqrt( devProp.maxThreadsPerBlock ));
	int N_sm				= devProp.multiProcessorCount;
	int max_threads_per_SM	= devProp.maxThreadsPerMultiProcessor;


	/* ....::: ALLOCATION :::.... */
	// -0- read metadata
	MDbin					= geotiffinfo( FIL_BIN, 1 );
	MDroi 					= geotiffinfo( FIL_ROI, 1 );
	MDuint 					= MDbin;
	MDuint.pixel_type 		= GDT_UInt32;
	map_len 				= MDbin.width*MDbin.heigth;
	// MANIPULATION:
	unsigned int tiledimX 	= sqrt_nmax_threads;
	unsigned int tiledimY 	= sqrt_nmax_threads;
	unsigned int WIDTH 		= MDbin.width;
	unsigned int HEIGHT 	= MDbin.heigth;
	// I need to add a first row of zeros, to let the kernels work fine also at location (0,0),
	// where the LABEL cannot be different from zero (the LABEL is equal to thread absolute position).
	unsigned int HEIGHT_1 	= HEIGHT +1;
	// X-dir of extended array
	unsigned int ntilesX 	= ceil( (double)(WIDTH-1) / (double)(tiledimX-1)  );
	unsigned int ntX_less	= floor( (double)(WIDTH-1) / (double)(tiledimX-1)  );
	unsigned int WIDTH_e	= ( ntilesX-ntX_less ) + ( ntX_less*tiledimX ) + ( WIDTH-1 -ntX_less*(tiledimX-1) );
	// Y-dir of extended array
	unsigned int ntilesY	= ceil( (double)(HEIGHT_1-1) / (double)(tiledimY-1)  );
	unsigned int ntY_less	= floor( (double)(HEIGHT_1-1) / (double)(tiledimY-1)  );
	unsigned int HEIGHT_e	= ( ntilesY-ntY_less ) + ( ntY_less*tiledimY ) + ( HEIGHT_1-1 -ntY_less*(tiledimY-1) );
	// size of arrays
	size_t sizeChar  		= WIDTH   * HEIGHT   * sizeof(unsigned char); // it does not need the offset
	size_t sizeChar_o  		= WIDTH   * HEIGHT_1 * sizeof(unsigned char); // it accounts for the offset
//	size_t sizeUintL 		= WIDTH   * HEIGHT_1 * sizeof(unsigned int);
	size_t sizeUintL_s 		= WIDTH   * HEIGHT   * sizeof(unsigned int);
	size_t sizeUintL_e 		= WIDTH_e * HEIGHT_e * sizeof(unsigned int);  // the offset is considered (using HEIGHT_1 to define HEIGHT_e)
	size_t sizeBins 		= ntilesX*ntilesY*sizeof(unsigned int);

	// -1- load geotiff [urban_cpu]
	unsigned char *urban_cpu	= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *ROI 			= (unsigned char *) CPLMalloc( sizeChar );
	printf("Importing...\t%s\t",FIL_BIN);
	geotiffread( FIL_BIN,MDbin,urban_cpu );
	printf("...done!\n");
	printf("Importing...\t%s\t",FIL_ROI);
	geotiffread( FIL_ROI, MDroi, &ROI[0] );
	printf("...done!\n");

	// -2- urban_gpu -- stream[0]
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&urban_gpu, 	sizeChar_o 	) );
	CUDA_CHECK_RETURN( cudaMemset( urban_gpu, 0, 			sizeChar_o 	) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&dev_ROI, 		sizeChar_o 	) );
	// I consider the WIDTH offset in copy to leave the first row with all zeros!
	CUDA_CHECK_RETURN( cudaMemcpy( urban_gpu+WIDTH,urban_cpu,sizeChar, cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN( cudaMemcpy( dev_ROI, ROI, 			 sizeChar, cudaMemcpyHostToDevice ) );
	// -3- lab_mat_cpu
	CUDA_CHECK_RETURN( cudaMallocHost(&lab_mat_cpu,	 		sizeUintL_e	) );
	CUDA_CHECK_RETURN( cudaMallocHost(&lab_mat_cpu_f,		sizeUintL_s	) );
	// -4- lab_mat_gpu  -- stream[1]
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&lab_mat_gpu, 	sizeUintL_e ) );
	CUDA_CHECK_RETURN( cudaMemset( lab_mat_gpu, 0, 			sizeUintL_e ) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&lab_mat_gpu_f, sizeUintL_s ) );
	CUDA_CHECK_RETURN( cudaMemset( lab_mat_gpu_f,0, 		sizeUintL_s ) );
	// -5- bins
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&bins_gpu, 		sizeBins 	) );
	CUDA_CHECK_RETURN( cudaMemset( bins_gpu, 0, 			sizeBins 	) );
	CUDA_CHECK_RETURN( cudaMallocHost((void**)&bins_cpu,	sizeBins 	) );
	CUDA_CHECK_RETURN( cudaMallocHost((void**)&cumsum,		sizeBins 	) );
	// -6- ones
//	CUDA_CHECK_RETURN( cudaMalloc( (void **)&ones_gpu, 		sizeUintL_e ) );
//	CUDA_CHECK_RETURN( cudaMemset( ones_gpu, 1,				sizeUintL_e ) );
	/* ....::: ALLOCATION :::.... */

/*
 *		KERNELS INVOCATION
 *
 *			*************************
 *			-1- intra_tile_labeling		|  --> 1st Stage :: intra-tile		:: mandatory
 *
 *			-2- stitching_tiles			|\
 *			-3- root_equivalence		|_|--> 2nd Stage :: inter-tiles		:: mandatory
 *
 *			-4- intra_tile_re_label		|  --> 3rd Stage :: intra-tile		:: mandatory
 *
 *			-5- count_labels			|\
 *			-6- labels__1_to_N			| |--> 4th Stage :: labels 1 to N	:: optional (set relabel)
 *			-7- intratile_relabel_1toN	|/
 *
 *			-8- del_duplicated_lines	|  --> 5th Stage :: adjust size		:: mandatory
 *
 *			*************************
 */

	dim3 	block(tiledimX,tiledimY,1);
	dim3 	grid(ntilesX,ntilesY,1);
	unsigned int 	sh_mem	= (tiledimX*tiledimY)*(sizeof(unsigned int));
	dim3 	block_2(tiledimX,1,1); // ==> this is only possible if the block is squared !!!!!!!! Because I use the same threads for cols & rows
	dim3 	grid_2(ntilesX,ntilesY,1);
	dim3 	block_3(tiledimX*tiledimY,1,1);
	dim3 	grid_3(ntilesX*ntilesY,1,1);
	// reduce6

	/* ....::: [1/5 stage] INTRA-TILE :::.... */
	start_t = clock();
	intra_tile_labeling<<<grid,block,sh_mem>>>(urban_gpu,WIDTH,HEIGHT_1,WIDTH_e,HEIGHT_e,lab_mat_gpu);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_1,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		CUDA_CHECK_RETURN( cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL_e,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_1);
		write_labmat_tiled(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
		sprintf(buffer,"%s/data/-%d-%s__k1.txt",BASE_PATH,count_print,kern_1);
		write_labmat_full(lab_mat_cpu, HEIGHT_e, WIDTH_e, buffer);
		//write_labmat_tiled_without_duplicated_LINES(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [1/5 stage] :::.... */

	/* ....::: [2/5 stage] STITCHING :::.... */
	start_t = clock();
	stitching_tiles<NTHREADSX><<<grid,block_2>>>(lab_mat_gpu,tiledimY, WIDTH_e, HEIGHT_e);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_2,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		CUDA_CHECK_RETURN( cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL_e,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_2);
		write_labmat_tiled(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
		sprintf(buffer,"%s/data/-%d-%s__k2.txt",BASE_PATH,count_print,kern_2);
		write_labmat_full(lab_mat_cpu, HEIGHT_e, WIDTH_e, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	start_t = clock();
	root_equivalence<NTHREADSX><<<grid_2,block_2>>>(lab_mat_gpu,tiledimY, WIDTH_e, HEIGHT_e);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_3,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		CUDA_CHECK_RETURN( cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL_e,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_3);
		write_labmat_tiled(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
		sprintf(buffer,"%s/data/-%d-%s__k3.txt",BASE_PATH,count_print,kern_3);
		write_labmat_full(lab_mat_cpu, HEIGHT_e, WIDTH_e, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [2/5 stage] :::.... */

	/* ....::: [3/5 stage] RE-LABELING :::.... */
	start_t = clock();
	intra_tile_re_label<<<grid,block,sh_mem>>>(WIDTH_e,HEIGHT_e,lab_mat_gpu);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_4,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		CUDA_CHECK_RETURN( cudaMemcpy( lab_mat_cpu,lab_mat_gpu,	sizeUintL_e,cudaMemcpyDeviceToHost ) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_4);
		write_labmat_tiled(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [3/5 stage] :::.... */

if (relabel){
	/* ....::: [4/5 stage] DEL DUPLICATES :::.... */
	start_t = clock();
	count_labels<<<grid,block,sh_mem>>>(WIDTH_e,HEIGHT_e,lab_mat_gpu,bins_gpu);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	CUDA_CHECK_RETURN( cudaMemcpy( bins_cpu, bins_gpu, sizeBins, cudaMemcpyDeviceToHost ) );
	cumsum[0] = 0;
	Nbins = bins_cpu[0];
	//printf("%4s %12s %12s\n", "ii", "bins[ii]", "cumsum[ii]" );
	//printf("%4d %12d %12d\n", 0, bins_cpu[0], cumsum[0] );
	for(ii=1;ii<ntilesX*ntilesY;ii++){
		cumsum[ii] = Nbins;
		Nbins += bins_cpu[ii];
		//cumsum[ii] = cumsum[ii-1] + bins_cpu[ii-1];
		//if(bins_cpu[ii]!=0) printf("%4d %12d %12d\n", ii, bins_cpu[ii], cumsum[ii] );
	}
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_4_a,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	start_t = clock();
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&ID_rand_gpu,Nbins*sizeof(unsigned int) ) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&ID_1toN_gpu,Nbins*sizeof(unsigned int) ) );
	CUDA_CHECK_RETURN( cudaMemcpy( bins_gpu, cumsum, sizeBins, cudaMemcpyHostToDevice ) );
	unsigned int bdx_e 	= WIDTH_e  - (ntilesX-1)*tiledimX;
	//unsigned int kmax_e = bins_cpu[ntilesX*ntilesY-1];
	unsigned int kmax_e = Nbins;//cumsum[ntilesX*ntilesY-1];
	labels__1_to_N<<<grid,block,sh_mem>>>( WIDTH_e, HEIGHT_e, lab_mat_gpu, bins_gpu, kmax_e, bdx_e, ID_rand_gpu, ID_1toN_gpu );
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_4_b,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		//tmp code :: save as GeoTiff
		del_duplicated_lines<<<grid,block>>>(lab_mat_gpu,WIDTH_e,HEIGHT_e, lab_mat_gpu_f,WIDTH,HEIGHT);
		CUDA_CHECK_RETURN( cudaMemcpy( lab_mat_cpu_f,lab_mat_gpu_f,	sizeUintL_s,cudaMemcpyDeviceToHost ) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_4_b);
		geotiffwrite(FIL_BIN,buffer,MDuint,lab_mat_cpu_f);
		/*
		CUDA_CHECK_RETURN( cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL_e,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_4_b);
		write_labmat_tiled(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
		sprintf(buffer,"%s/data/-%d-%s__k3.txt",BASE_PATH,count_print,kern_4_b);
		write_labmat_full(lab_mat_cpu, HEIGHT_e, WIDTH_e, buffer);
		*/
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	start_t = clock();
	//intratile_relabel_1toN_notgood<<<grid,block,sh_mem>>>(	WIDTH_e, HEIGHT_e, lab_mat_gpu, bins_gpu, Nbins, ID_rand_gpu, ID_1toN_gpu );
	intratile_relabel_1toN<<<grid,block,sh_mem>>>(WIDTH_e,HEIGHT_e,lab_mat_gpu);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_4_c,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		CUDA_CHECK_RETURN( cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL_e,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_4_c);
		write_labmat_tiled(lab_mat_cpu, tiledimY,tiledimY, ntilesY,ntilesX, HEIGHT_e,WIDTH_e, buffer);
		sprintf(buffer,"%s/data/-%d-%s__k3.txt",BASE_PATH,count_print,kern_4_c);
		write_labmat_full(lab_mat_cpu, HEIGHT_e, WIDTH_e, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [4/5 stage] :::.... */
}

	/* ....::: [5/5 stage] :::.... */
	start_t = clock();
	del_duplicated_lines<<<grid,block>>>(lab_mat_gpu,WIDTH_e,HEIGHT_e, lab_mat_gpu_f,WIDTH,HEIGHT);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_5,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	CUDA_CHECK_RETURN( cudaMemcpy( lab_mat_cpu_f,lab_mat_gpu_f,	sizeUintL_s,cudaMemcpyDeviceToHost ) );
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (1){
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_5);
		//write_labmat_full(lab_mat_cpu_f, HEIGHT, WIDTH, buffer);
		geotiffwrite(FIL_BIN,FIL_LAB,MDuint,lab_mat_cpu_f);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [5/5 stage] :::.... */

if (relabel){
	/* ....::: [6/5 stage] :::.... */
	// -7- histogram
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void **)&h_histogram,	Nbins*sizeof( unsigned int )) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&d_histogram,  Nbins*sizeof( unsigned int )) );
	CUDA_CHECK_RETURN( cudaMemset( 		d_histogram, 0, 		Nbins*sizeof( unsigned int )) );

	int BLOCK_DIM 		= 512;
	num_blocks_per_SM	= max_threads_per_SM / BLOCK_DIM;// e.g. 1536/512 = 3
	mapel_per_thread    = (unsigned int)ceil( (double)map_len / (double)((BLOCK_DIM)*N_sm*num_blocks_per_SM) );// e.g. n / (14*3*512*2)
	dim3 	dimBlock( threads, 1, 1 );
	dim3 	dimGrid(  N_sm*num_blocks_per_SM,  1, 1 );
	int smemSize 		= (Nbins+1) * sizeof(unsigned int);// sdata=threads*Nbins is allocated dinamically, while sdata_j=threads*1 and is allocated statically
	start_t = clock();
	// I/O config of reduce6_hist ––> (*g_idata, *ROI, *g_ohist, map_len, mapel_per_thread, Nbins)
/*	if (isPow2(map_len)){ reduce6_hist<unsigned int, 512, true> <<< dimGrid, dimBlock, smemSize >>>(lab_mat_gpu_f, dev_ROI, d_histogram, map_len, Nbins);
	}else{	 			  reduce6_hist<unsigned int, 512, false><<< dimGrid, dimBlock, smemSize >>>(lab_mat_gpu_f, dev_ROI, d_histogram, map_len, Nbins);}
*/	reduce6_hist<unsigned int> <<< dimGrid, dimBlock, smemSize >>>(lab_mat_gpu_f, dev_ROI, d_histogram, map_len, Nbins);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_6,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	CUDA_CHECK_RETURN( cudaMemcpy(h_histogram,d_histogram,	(size_t)Nbins*sizeof( unsigned int ),cudaMemcpyDeviceToHost) );
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [6/5 stage] :::.... */
}

	/* DO NOT EDIT THE FOLLOWING PRINT (it's used in MatLab to catch the elapsed time!)*/
	//printf("Total time: %f [msec]\n", (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000 );
	printf("_______________________________________________\n");
	printf("  %24s\t%6d [msec]\n", "Total time:",elapsed_time );

	// SAVE lab_mat to file and compare with MatLab
	//write_labmat_matlab(lab_mat_cpu, tiledimX, tiledimY, ntilesX, ntilesY, Lcuda);
	geotiffwrite(FIL_BIN,FIL_LAB,MDuint,lab_mat_cpu_f);
	// SAVE histogram
	sprintf(buffer,"%s/data/%s.txt",BASE_PATH,"cu_histogram");
	write_labmat_full(h_histogram, Nbins+1, 1, buffer);

	FILE *fid;
	sprintf(buffer,"%s/data/%s.txt",BASE_PATH,"performance");
	fid = fopen(buffer,"a");
	if (fid == NULL) { printf("Error opening file %s!\n",buffer); exit(1); }
	fprintf(fid,"Image[%d,%d], tiledim[%d,%d], ntiles[%d,%d]\n",HEIGHT,WIDTH,tiledimX,tiledimY,ntilesX,ntilesY);
	fprintf(fid,"%s: %d\n","Elapsed time",elapsed_time);
	fclose(fid);

	// FREE MEMORY:
	cudaFreeHost(lab_mat_cpu_f);
	cudaFreeHost(lab_mat_cpu);
	cudaFreeHost(urban_cpu);
	cudaFree(lab_mat_gpu_f);
	cudaFree(lab_mat_gpu);
	cudaFree(urban_gpu);

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	printf("\n\n%s\n", "Finished!!");

	// RETURN:
	return 0;
}
