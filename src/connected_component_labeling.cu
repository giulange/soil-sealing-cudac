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
	return (blockIdx.x < gridDim.x-1) ? blockDim.x : WIDTH - (gridDim.x-1)*blockDim.x;
}
__device__ unsigned int fBDY(unsigned int HEIGHT){ // NOTE: I'm assuming that blockDim.x = blockDim.y
	return (blockIdx.y < gridDim.y-1) ? blockDim.x : HEIGHT - (gridDim.y-1)*blockDim.x;
}
__device__ unsigned int fBDY_cross(unsigned int HEIGHT){ // NOTE: I'm assuming that blockDim.x = blockDim.y
	return (blockIdx.y < gridDim.y-1) ? blockDim.x : HEIGHT - (gridDim.y-1)*blockDim.x;
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
#define			Vo			1	// object value
#define			Vb			0	// object value
char			buffer[255];
const char 		*BASE_PATH	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing";
// I/-
//const char 		*FIL_BIN	= "/home/giuliano/git/cuda/fragmentation/data/BIN.tif";
//const char 		*FIL_BIN	= "/home/giuliano/git/cuda/fragmentation/data/BIN-cropped.tif";
const char		*FIL_BIN	= "/home/giuliano/git/cuda/fragmentation/data/imp_mosaic_char_2006_cropped.tif";
// -/O
const char		*Lcuda		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/CUDA-code.txt";
const char 		*FIL_LAB 	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/LAB-MAT-cuda.tif";
// kernel_names
const char 		*kern_1 	= "intra_tile_labeling";
const char 		*kern_2 	= "stitching_tiles";
const char 		*kern_3 	= "root_equivalence";
const char 		*kern_4 	= "intra_tile_re_label";
const char 		*kern_5 	= "del_duplicated_lines";

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

void read_urbmat(unsigned char *urban, unsigned int nrows, unsigned int ncols, const char *filename)
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

void write_labmat_tiled(	unsigned int *lab_mat,
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
void write_labmat_tiled_without_duplicated_LINES(
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
void write_labmat_full(unsigned int *lab_mat, unsigned int HEIGHT, unsigned int WIDTH, const char *filename)
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

void write_labmat_matlab(unsigned int *lab_mat, unsigned int nr, unsigned int nc, unsigned int ntilesX, unsigned int ntilesY, const char *filename)
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

__global__ void intra_tile_labeling(const unsigned char *urban,unsigned int WIDTH,unsigned int HEIGHT,unsigned int WIDTH_e,unsigned int HEIGHT_e,unsigned int *lab_mat)
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
__global__ void stitching_tiles(	unsigned int *lab_mat,
									const unsigned int bdy,
									const unsigned int WIDTH_e,
									const unsigned int HEIGHT_e)
{
	/*
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
__global__ void root_equivalence(	unsigned int *lab_mat,
									const unsigned int bdy,
									const unsigned int WIDTH_e,
									const unsigned int HEIGHT_e		)
{

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

__global__ void intra_tile_re_label(unsigned int WIDTH_e, unsigned int HEIGHT_e, unsigned int *lab_mat)
{
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
		// try to write a sequence of IDs starting from 1 to N found labels!!
		// ...some code...
		if  (lab_mat[ttid]!=Vb)  lab_mat[ttid]=lab_mat[lab_mat[ttid]];
	}
}

__global__ void
del_duplicated_lines( 	const unsigned int *lab_mat_gpu,	unsigned int WIDTH_e,unsigned int HEIGHT_e,
							  unsigned int *lab_mat_gpu_f,	unsigned int WIDTH,	 unsigned int HEIGHT	){

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

int main(int argc, char **argv){

	// Parameter:
	const unsigned int 	NTHREADSX 			= 32;

	// DECLARATIONS:
	bool 				printme				= false;
	unsigned int  		*lab_mat_cpu, *lab_mat_gpu, *lab_mat_cpu_f, *lab_mat_gpu_f;
	unsigned char 		*urban_gpu;
	metadata 			MDbin,MDuint;
	unsigned int		gpuDev				= 0;
	unsigned int 		sqrt_nmax_threads 	= 0;
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

	/* ....::: ALLOCATION :::.... */
	// -0- read metadata
	MDbin					= geotiffinfo( FIL_BIN, 1 );
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

	// -1- load geotiff [urban_cpu]
	unsigned char *urban_cpu	= (unsigned char *) CPLMalloc( sizeChar );
	printf("Importing...\t%s\n\n",FIL_BIN);
	geotiffread(FIL_BIN,MDbin,urban_cpu);

	// -2- urban_gpu -- stream[0]
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&urban_gpu, sizeChar_o ) );
	CUDA_CHECK_RETURN( cudaMemset( urban_gpu,0, sizeChar_o ) );
	// I consider the WIDTH offset in copy to leave the first row with all zeros!
	CUDA_CHECK_RETURN( cudaMemcpy( urban_gpu+WIDTH,urban_cpu,	sizeChar,cudaMemcpyHostToDevice ) );
	// -3- lab_mat_cpu
	CUDA_CHECK_RETURN( cudaMallocHost(&lab_mat_cpu,	 sizeUintL_e) );
	CUDA_CHECK_RETURN( cudaMallocHost(&lab_mat_cpu_f,sizeUintL_s) );
	// -4- lab_mat_gpu  -- stream[1]
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&lab_mat_gpu, sizeUintL_e ) );
	CUDA_CHECK_RETURN( cudaMemset( lab_mat_gpu,0, sizeUintL_e ) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&lab_mat_gpu_f, sizeUintL_s ) );
	CUDA_CHECK_RETURN( cudaMemset( lab_mat_gpu_f,0, sizeUintL_s ) );
	/* ....::: ALLOCATION :::.... */

/*
 *		KERNELS INVOCATION
 *
 *			*************************
 *			-1- intra_tile_labeling		| --> 1st Stage
 *
 *			-2- stitching_tiles			|\
 *			-2- root_equivalence		| --> 2nd Stage
 *
 *			-3- intra_tile_re_label		| --> 3rd Stage
 *
 *			-4- del_duplicated_lines	| --> 4th Stage
 *			*************************
 */

	dim3 	block(tiledimX,tiledimY,1);
	dim3 	grid(ntilesX,ntilesY,1);
	unsigned int 	sh_mem	= (tiledimX*tiledimY)*(sizeof(unsigned int));
	dim3 	block_2(tiledimX,1,1); // ==> this is only possible if the block is squared !!!!!!!! Because I use the same threads for cols & rows
	dim3 	grid_2(ntilesX,ntilesY,1);

	/* ....::: [1/4 stage] INTRA-TILE :::.... */
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
	/* ....::: [1/4 stage] :::.... */

	/* ....::: [2/4 stage] STITCHING :::.... */
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
		sprintf(buffer,"%s/data/-%d-%s__k2.txt",BASE_PATH,count_print,kern_1);
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
		sprintf(buffer,"%s/data/-%d-%s__k3.txt",BASE_PATH,count_print,kern_1);
		write_labmat_full(lab_mat_cpu, HEIGHT_e, WIDTH_e, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [2/4 stage] :::.... */

	/* ....::: [3/4 stage] RE-LABELING :::.... */
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
	/* ....::: [3/4 stage] :::.... */

	/* ....::: [4/4 stage] DEL DUPLICATES :::.... */
	start_t = clock();
	del_duplicated_lines<<<grid,block>>>(lab_mat_gpu,WIDTH_e,HEIGHT_e, lab_mat_gpu_f,WIDTH,HEIGHT);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %20s\t%6d [msec]\n",++count_print,kern_5,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	CUDA_CHECK_RETURN( cudaMemcpy( lab_mat_cpu_f,lab_mat_gpu_f,	sizeUintL_s,cudaMemcpyDeviceToHost ) );
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_5);
		write_labmat_full(lab_mat_cpu_f, HEIGHT, WIDTH, buffer);
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	/* ....::: [4/4 stage] :::.... */

	/* DO NOT EDIT THE FOLLOWING PRINT (it's used in MatLab to catch the elapsed time!)*/
	//printf("Total time: %f [msec]\n", (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000 );
	printf("_______________________________________________\n");
	printf("  %24s\t%6d [msec]\n", "Total time:",elapsed_time );

	// SAVE lab_mat to file and compare with MatLab
	MDuint 						= MDbin;
	MDuint.pixel_type 			= GDT_UInt32;
	//write_labmat_matlab(lab_mat_cpu, tiledimX, tiledimY, ntilesX, ntilesY, Lcuda);
	geotiffwrite(FIL_BIN,FIL_LAB,MDuint,lab_mat_cpu_f);

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
