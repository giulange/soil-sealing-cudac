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

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

//	-indexes
#define durban(cc,rr)	urban[		(cc)	+	(rr)	*(blockDim.x)	] // I: scan value at current [r,c]
#define nw_pol(cc,rr)	lab_mat_sh[	(cc-1)	+	(rr-1)	*(blockDim.x)	] // O: scan value at North-West
#define nn_pol(cc,rr)	lab_mat_sh[	(cc+0)	+	(rr-1)	*(blockDim.x)	] // O: scan value at North
#define ne_pol(cc,rr)	lab_mat_sh[	(cc+1)	+	(rr-1)	*(blockDim.x)	] // O: scan value at North-East
#define ww_pol(cc,rr)	lab_mat_sh[	(cc-1)	+	(rr+0)	*(blockDim.x)	] // O: scan value at West
#define ee_pol(cc,rr)	lab_mat_sh[	(cc+1)	+	(rr+0)	*(blockDim.x)	] // O: scan value at West
#define sw_pol(cc,rr)	lab_mat_sh[	(cc-1)	+	(rr+1)	*(blockDim.x)	] // O: scan value at South-West
#define ss_pol(cc,rr)	lab_mat_sh[	(cc+0)	+	(rr+1)	*(blockDim.x)	] // O: scan value at South-West
#define se_pol(cc,rr)	lab_mat_sh[	(cc+1)	+	(rr+1)	*(blockDim.x)	] // O: scan value at South-West
#define cc_pol(cc,rr)	lab_mat_sh[	(cc+0)	+	(rr+0)	*(blockDim.x)	] // O: scan value at current [r,c] which is shifted by [1,1] in O

// GLOBAL VARIABLES
#define			Vo			1	// object value
#define			Vb			0	// object value
char			buffer[255];
const char		*Lcuda		= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/CUDA-code.txt";
const char 		*ALL_txt	= "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/ALL.txt";

//---------------------------- FUNCTIONS PROTOTYPES
//		** I/O **
void read_urbmat(unsigned char *, unsigned int, unsigned int, const char *);
void write_labmat_tiled( unsigned int *,  unsigned int, unsigned int, unsigned int, unsigned int, const char *);
void write_labmat_matlab(unsigned int *,  unsigned int, unsigned int, unsigned int, unsigned int, const char *);
//		** kernels **
//	(1)
__global__ void intra_tile_labeling( const unsigned char *,unsigned int, unsigned int * );
//	(2)
__global__ void stitching_tiles( unsigned int *,const unsigned int,const unsigned int );
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
			fscanf(fid, "%d",&a);
			urban[cc+rr*ncols]=(unsigned char)a;
			//printf("%d ",a);
		}
		//printf("\n");
	}
	fclose(fid);
}

void write_labmat_tiled(unsigned int *lab_mat, unsigned int nr, unsigned int nc, unsigned int ntilesX, unsigned int ntilesY, const char *filename)
{
	unsigned int rr,cc,ntX,ntY;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }
	int offset;

	for(ntY=0;ntY<ntilesY;ntY++)
	{
		for(rr=0;rr<nr;rr++)
		{
			for(ntX=0;ntX<ntilesX;ntX++)
			{
				for(cc=0;cc<nc;cc++)
				{
					/*if( !(((cc==nc-1) && ((ntilesX*ntY+ntX+1)%ntilesX)==0))	&&	// do not print last column
						!(((rr==nr-1) && (ntY==ntilesY-1))) 					// do not print last row
					)*/
					{
						offset = (ntilesX*ntY+ntX)*nc*nr+(nc*rr+cc);
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

void write_labmat_matlab(unsigned int *lab_mat, unsigned int nr, unsigned int nc, unsigned int ntilesX, unsigned int ntilesY, const char *filename)
{
	unsigned int rr,cc,ntX,ntY;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }
	int offset;

	for(ntY=0;ntY<ntilesY;ntY++)
	{
		for(rr=1;rr<nr;rr++)
		{
			for(ntX=0;ntX<ntilesX;ntX++)
			{
				for(cc=1;cc<nc;cc++)
				{
					if( !(((cc==nc-1) && ((ntilesX*ntY+ntX+1)%ntilesX)==0))	&&	// do not print last column
						!(((rr==nr-1) && (ntY==ntilesY-1))) 					// do not print last row

					)
					{
						offset = (ntilesX*ntY+ntX)*nc*nr+(nc*rr+cc);
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

__global__ void intra_tile_labeling(const unsigned char *urban,unsigned int NC,unsigned int *lab_mat)
{
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
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int iTile		= gdx * biy + bix;

	unsigned int otid		= bdx * r + c;
	unsigned int itid		= (r * NC + c) 					+		// dentro la 1° tile		0
							  (bdx - 1) * (iTile % gdx)		+		// itile in orizzontale		0
							  (iTile / gdx) * (bdy-1) * NC;			// itile in verticale		0
	unsigned int ttid 		= iTile*bdx*bdy+otid;
/*	unsigned int stid		= (r * gdx * bdx + c) 			+		// dentro la 1° tile		0
			  	  	  	  	  (bdx - 0) * (iTile % gdx)		+		// itile in orizzontale		0
			  	  	  	  	  (iTile / gdx) * (bdy-0) * gdx * bdx;	// itile in verticale		42
*/
	if (iTile<gdx*gdy)
	{
		lab_mat_sh[otid] 	= 0;
		// if (r,c) is object pixel
		//if  (urban[ttid]==Vo)  lab_mat_sh[otid] = ttid; // use ttid with 	"linearize_tiles"
		if  (urban[itid]==Vo)  lab_mat_sh[otid] = ttid; // use itid without "linearize_tiles"
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
			if(	c>0 && r>0 && nw_pol(c,r)!=0 && nw_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = nw_pol(c,r); found = true; }
			// NN:
			if( r>0 && nn_pol(c,r)!=0 && nn_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = nn_pol(c,r); found = true; }
			// NE:
			if( c<bdx-1 && r>0 && ne_pol(c,r)!=0 && ne_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ne_pol(c,r); found = true; }
			// WW:
			if( c>0 && ww_pol(c,r)!=0 && ww_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ww_pol(c,r); found = true; }
			// EE:
			if( c<bdx-1 && ee_pol(c,r)!=0 && ee_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ee_pol(c,r); found = true; }
			// SW:
			if( c>0 && r<bdy-1 && sw_pol(c,r)!=0 && sw_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = sw_pol(c,r); found = true; }
			// SS:
			if( r<bdy-1 && ss_pol(c,r)!=0 && ss_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ss_pol(c,r); found = true; }
			// SE:
			if( c<bdx-1 && r<bdy-1 && se_pol(c,r)!=0 && se_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = se_pol(c,r); found = true; }

			__syncthreads();
		}

		/*
		 * 	To linearize I write using ttid.
		 * 	To leave same matrix configuration as input urban use stid instead!!
		 */
		lab_mat[ttid] = lab_mat_sh[otid];
		//__syncthreads();
	}
}

template <unsigned int NTHREADSX>
__global__ void stitching_tiles(	unsigned int *lab_mat,
									const unsigned int tiledimX,
									const unsigned int tiledimY		)
{
	/*
	 * 	NOTE:
	 * 		> xx_yy is the tile xx and border yy (e.g. nn_ss is tile at north and border at south).
	 */

	//unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	//unsigned int bdx		= blockDim.x;
	//unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int iTile		= gdx * biy + bix;

	// TILES:
	int nTiles			= gdx * gdy;
	int nn_tile			= (iTile < gdx)?-1:(iTile - gdx);		// nn tile of cc tile
	int ww_tile			= ((iTile % gdx)==0)?-1:iTile-1;		// ww tile of cc tile

	// SIDES:
	int c_nn_tid		=	c 								+	// (2) within-tile
							tiledimX*tiledimY * iTile;			// (1) horizontal offset

	int nn_tid			= 	c +	tiledimX*(tiledimY-1) 		+	// (2) within-tile
							tiledimX*tiledimY * nn_tile;		// (1) horizontal offset

	int c_ww_tid		=	c*tiledimX 						+	// (2) within-tile
							tiledimX*tiledimY * iTile;			// (1) horizontal offset

	int ww_tid			= 	(c+1)*tiledimX-1 				+	// (2) within-tile
							tiledimX*tiledimY * ww_tile;		// (1) horizontal offset

	// SHARED: "tile_border" ==> cc_nn is border North of Center tile
	__shared__ unsigned int cc_nn[NTHREADSX];
	__shared__ unsigned int nn_ss[NTHREADSX];
	__shared__ unsigned int cc_ww[NTHREADSX];
	__shared__ unsigned int ww_ee[NTHREADSX];
	__shared__ unsigned int __old[NTHREADSX];
	__shared__ unsigned int _min_[NTHREADSX];
	__shared__ unsigned int _max_[NTHREADSX];

	if( iTile < nTiles )
	{
		// ...::NORTH::...
		if( nn_tile>=0 ){
			//recursion ( lab_mat, c_nn_tid, nn_tid );
			/*
			 * 		(1) **list** { cc_nn(i), nn_ss(i) }
			 */
			cc_nn[ c ]		= lab_mat[ c_nn_tid ];
			nn_ss[ c ] 		= lab_mat[ nn_tid ];
			__syncthreads();
			/*
			 * 		(2) **recursion applying split-rules**
			 */
			__old[ c ] = atomicMin( &lab_mat[ cc_nn[c] ], nn_ss[ c ] ); // write the current min val where the index cc_nn[c] is in lab_mat.
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
		if( ww_tile>=0 ){
			//recursion ( lab_mat, c_ww_tid, ww_tid );
			/*
			 * 		(1) **list** { cc_nn(i), nn_ss(i) }
			 */
			cc_ww[ c ]		= lab_mat[ c_ww_tid ];
			ww_ee[ c ] 		= lab_mat[ ww_tid ];
			__syncthreads();

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
}

template <unsigned int NTHREADSX>
__global__ void root_equivalence(	unsigned int *lab_mat,
									const unsigned int tiledimX,
									const unsigned int tiledimY		)
{

	//unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	//unsigned int bdx		= blockDim.x;
	//unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int iTile		= gdx * biy + bix;

	// TILES:
	int nTiles			= gdx * gdy;
	int nn_tile			= (iTile < gdx)?-1:(iTile - gdx);			// nn tile of cc tile
	int ww_tile			= ((iTile % gdx)==0)?-1:iTile-1;			// ww tile of cc tile
	int ss_tile			= (iTile >= gdx*(gdy-1))?-1:(iTile + gdx);	// tile index of ss
	int ee_tile			= ((iTile % gdx)==gdx-1)?-1:iTile+1;		// tile index of ee

	// SIDES:
	int c_nn_tid		=	c 								+		// (2) within-tile
							tiledimX*tiledimY * iTile;				// (1) horizontal offset
	int nn_tid			= 	c + tiledimX*(tiledimY-1)		+		// (2) within-tile
							tiledimX*tiledimY * nn_tile;			// (1) horizontal offset
	int c_ww_tid		=	c*tiledimX 						+		// (2) within-tile
							tiledimX*tiledimY * iTile;				// (1) horizontal offset
	int ww_tid			= 	(c+1)*tiledimX-1 				+		// (2) within-tile
							tiledimX*tiledimY * ww_tile;			// (1) horizontal offset

	int c_ss_tid		=	c + tiledimX*(tiledimY-1)		+		// (2) within-tile
							tiledimX*tiledimY * iTile;				// (1) horizontal offset
	int ss_tid			= 	c								+		// (2) within-tile
							tiledimX*tiledimY * ss_tile;			// (1) horizontal offset
	int c_ee_tid		=	(c+1)*tiledimX-1				+		// (2) within-tile
							tiledimX*tiledimY * iTile;				// (1) horizontal offset
	int ee_tid			= 	c*tiledimX 						+		// (2) within-tile
							tiledimX*tiledimY * ee_tile;			// (1) horizontal offset

	// SHARED:
	__shared__ unsigned int cc_nn[NTHREADSX];
	__shared__ unsigned int nn_ss[NTHREADSX];
	__shared__ unsigned int cc_ww[NTHREADSX];
	__shared__ unsigned int ww_ee[NTHREADSX];
	__shared__ unsigned int cc_ss[NTHREADSX];
	__shared__ unsigned int ss_nn[NTHREADSX];
	__shared__ unsigned int cc_ee[NTHREADSX];
	__shared__ unsigned int ee_ww[NTHREADSX];

	if( iTile < nTiles )
	{
		// ...::NORTH::...
		if( nn_tile>=0 ){
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
		if( ww_tile>=0 ){
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
		if( ss_tile>=0 ){
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
		if( ee_tile>=0 ){
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
}

__global__ void intra_tile_re_label(unsigned int NC,unsigned int *lab_mat)
{
	// See this link when using more then one extern __shared__ array:
	// 		http://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable
	//extern __shared__ unsigned char urban_sh[];
//	extern __shared__ unsigned int  lab_mat_sh[];

//	__shared__ bool found;

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int iTile		= gdx * biy + bix;

	unsigned int otid		= bdx * r + c;
/*	unsigned int itid		= (r * NC + c) 						+	// dentro la 1° tile		0
							  (bdx - 1) * (iTile % gdx)			+	// itile in orizzontale		0
							  (iTile / gdx) * (bdy-1) * NC;			// itile in verticale		0
*/	unsigned int ttid 		= iTile*bdx*bdy+otid;
/*	unsigned int stid		= (r * gdx * bdx + c) 		+			// dentro la 1° tile		0
			  	  	  	  	  (bdx - 0) * (iTile % gdx)			+	// itile in orizzontale		0
			  	  	  	  	  (iTile / gdx) * (bdy-0) * gdx * bdx;	// itile in verticale		42
*/
	if (iTile<gdx*gdy)
	{
		// try to write a sequence of IDs starting from 1 to N found labels!!
		// ...some code...

		if  (lab_mat[ttid]!=Vb)  lab_mat[ttid]=lab_mat[lab_mat[ttid]];
		//if  (urban[itid]==Vo)  urban[itid]=lab_mat[lab_mat[ttid]];
	}
}

int main(int argc, char **argv)
{
	// INPUTS
	unsigned int tiledimX 	= atoi( argv[1] );	// tile dim in X
	unsigned int tiledimY 	= atoi( argv[2] );	// tile dim in Y
	unsigned int NC1 		= atoi( argv[3] );	// ncols of imported image
	unsigned int NR1 		= atoi( argv[4] );	// nrows of imported image
	unsigned int printme	= atoi( argv[5] );	// print by-kernel output

	const unsigned int NTHREADSX = 32;			// how to let it be variable.

	if( NTHREADSX!=tiledimX ){
		fprintf(stderr, "Error: NTHREADSX(=%d) <> tiledimX(=%d)!\n", NTHREADSX,tiledimX);
		printf("\t[modify it according to tiledimX(=%d) (which should be equal to tiledimY(=%d)!!]\n\n",tiledimX,tiledimY);
		exit(EXIT_FAILURE);
	}

	// count the number of kernels that must print their LAB-MAT:
	unsigned int count_print=0;

	// MANIPULATION:
	// X dir
	unsigned int ntilesX 	= ceil( (double)(NC1+2-1) / (double)(tiledimX-1)  );
	unsigned int NC 		= ntilesX*(tiledimX-1) +1;// number of columns 	of URBAN with 1-pixel-width zero perimeter
	// Y dir
	unsigned int ntilesY	= ceil( (double)(NR1+2-1) / (double)(tiledimY-1)  );
	unsigned int NR 		= ntilesY*(tiledimY-1) +1;// number of rows		of URBAN with 1-pixel-width zero perimeter

	// DECLARATIONS:
	//	Error code to check return values for CUDA calls
	cudaError_t cudaLastErr = cudaSuccess;

	// size of arrays
	size_t sizeChar  		= NC*NR * sizeof(unsigned char);
	size_t sizeUintL 		= ntilesX*ntilesY*tiledimX*tiledimY * sizeof(unsigned int);
	// clocks:
	clock_t start_t, end_t;

	/* ....::: ALLOCATION :::.... */
	// -1- urban_cpu
	unsigned char *urban_cpu;
	cudaMallocHost(&urban_cpu,sizeChar);
	read_urbmat(urban_cpu, NR, NC, ALL_txt); // This function reads the Image and stores it in RAM with a 1-pixel-width zero-padding perimeter
	// -2- urban_gpu -- stream[0]
	unsigned char *urban_gpu;
	cudaLastErr = cudaMalloc( (void **)&urban_gpu, sizeChar );
	if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to allocate device array urban_gpu (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	//cudaLastErr = cudaMemsetAsync( urban_gpu,0, sizeChar, stream[0] );
	cudaLastErr = cudaMemset( urban_gpu,0, sizeChar );
	if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to set ZEROS in urban_gpu array on device (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	//cudaLastErr = cudaMemcpyAsync( urban_gpu,urban_cpu,	sizeChar,cudaMemcpyHostToDevice, stream[0] );
	cudaLastErr = cudaMemcpy( urban_gpu,urban_cpu,	sizeChar,cudaMemcpyHostToDevice );
	if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to copy array urban_cpu from host to device urban_gpu (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	// -3- lab_mat_cpu
	unsigned int  *lab_mat_cpu;
	cudaMallocHost(&lab_mat_cpu,sizeUintL);
	// -4- lab_mat_gpu  -- stream[1]
	unsigned int  *lab_mat_gpu;
	cudaLastErr = cudaMalloc( (void **)&lab_mat_gpu, sizeUintL );
	if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to allocate device array lab_mat_gpu (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	//cudaLastErr = cudaMemsetAsync( lab_mat_gpu,0, sizeUintL, stream[0] );
	cudaLastErr = cudaMemset( lab_mat_gpu,0, sizeUintL );
	if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to set ZEROS in lab_mat_gpu array on device (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	/* ....::: ALLOCATION :::.... */

	start_t = clock();
/*
 *		KERNELS INVOCATION
 *
 *			*************************
 *			-1- linearize_tiles [NO]	|\
 *			-2- intra_tile_labeling		| --> 1st Stage
 *
 *			-3- stitching_tiles			|\
 *			-4- root_equivalence		| --> 2nd Stage
 *
 *			-5- intra_tile_re_label		| --> 3rd Stage
 *			*************************
 */

	/* ....::: [1/3 stage] INTRA-TILE :::.... */
	dim3 	block(tiledimX,tiledimY,1);
	dim3 	grid(ntilesX,ntilesY,1);
	int 	sh_mem	= (tiledimX*tiledimY)*(sizeof(unsigned int)); // +sizeof(unsigned char)

	intra_tile_labeling<<<grid,block,sh_mem>>>(urban_gpu,NC,lab_mat_gpu);
	cudaLastErr 	= cudaGetLastError();
	if (cudaLastErr != cudaSuccess){ printf ("ERROR {intra_tile_labeling} -- %s\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		count_print++;
		printf("  -%d- %30s\n",count_print,"intra_tile_labeling");
		cudaLastErr 	= cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost);
		if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to allocate copy array lab_mat_gpu from device to host (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
		sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/-2-lab_mat_cpu-intra_tile_labeling.txt");
		write_labmat_tiled(lab_mat_cpu, tiledimY, tiledimX, ntilesX, ntilesY, buffer);
	}
	/* ....::: [1/3 stage] :::.... */

	/* ....::: [2/3 stage] STITCHING :::.... */
	dim3 	block_2(tiledimX,1,1);
	dim3 	grid_2(ntilesX,ntilesY,1);

	stitching_tiles<NTHREADSX><<<grid_2,block_2>>>(lab_mat_gpu,tiledimX,tiledimY);
	cudaLastErr 	= cudaGetLastError();
	if (cudaLastErr != cudaSuccess){ printf ("ERROR {stitching_tiles} -- %s\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		count_print++;
		printf("  -%d- %30s\n",count_print,"stitching_tiles");
		cudaLastErr 	= cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost);
		if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to allocate copy array lab_mat_gpu from device to host (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
		sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/-3-lab_mat_cpu-stitching_tiles.txt");
		write_labmat_tiled(lab_mat_cpu, tiledimY, tiledimX, ntilesX, ntilesY, buffer);
	}

	root_equivalence<NTHREADSX><<<grid_2,block_2>>>(lab_mat_gpu,tiledimX,tiledimY);
	cudaLastErr 	= cudaGetLastError();
	if (cudaLastErr != cudaSuccess){ printf ("ERROR {root_equivalence} -- %s\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		count_print++;
		printf("  -%d- %30s\n",count_print,"stitching_tiles");
		cudaLastErr 	= cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost);
		if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to allocate copy array lab_mat_gpu from device to host (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
		sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/-4-lab_mat_cpu-root_equivalence.txt");
		write_labmat_tiled(lab_mat_cpu, tiledimY, tiledimX, ntilesX, ntilesY, buffer);
	}
	/* ....::: [2/3 stage] :::.... */

	/* ....::: [3/3 stage] INTRA-TILE #2 :::.... */
	intra_tile_re_label<<<grid,block,sh_mem>>>(NC,lab_mat_gpu);
	cudaLastErr 	= cudaGetLastError();
	if (cudaLastErr != cudaSuccess){ printf ("ERROR {intra_tile_re_label} -- %s\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	// D2H --> [lab_mat_cpu]
	cudaLastErr 	= cudaMemcpy( lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost );
	if (cudaLastErr != cudaSuccess){ fprintf(stderr, "Failed to copy array lab_mat_gpu from device to host lab_mat_cpu (error code %s)!\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	/* INTERMEDIATE CHECK [activate/deactivate]*/
	if (printme){
		count_print++;
		printf("  -%d- %30s\n\n",count_print,"intra_tile_re_label");
		sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/-5-lab_mat_cpu-intra_tile_re_label.txt");
		write_labmat_tiled(lab_mat_cpu, tiledimY, tiledimX, ntilesX, ntilesY, buffer);
	}
	/* ....::: [3/3 stage] :::.... */
	end_t = clock();

	/* DO NOT EDIT THE FOLLOWING PRINT (it's used in MatLab to catch the elapsed time!)*/
	printf("Total time: %f [msec]\n", (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000 );

	// SAVE lab_mat to file and compare with MatLab
	sprintf(buffer,Lcuda);
	write_labmat_matlab(lab_mat_cpu, tiledimX, tiledimY, ntilesX, ntilesY, buffer);

	// FREE MEMORY:
	cudaFreeHost(lab_mat_cpu);
	cudaFreeHost(urban_cpu);
	cudaFree(lab_mat_gpu);
	cudaFree(urban_gpu);

	printf("\nFinished!!\n");
	// RETURN:
	return 0;
}
