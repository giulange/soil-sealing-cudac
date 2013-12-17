/*
	Object:		Raster-scan and label-equivalence-based algorithm.
	Authors:	Massimo Nicolazzo & Giuliano Langella
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
		> cc is is the background(=0)/foreground(=1) pixel at (r,c),
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
#include <helper_cuda.h>	// helper for checking cuda initialization and error checking

//	-indexes
#define durban(c,r)		urban[		(c)		+	(r)		*(blockDim.x)	] // I: scan value at current [r,c]
#define nw_pol(c,r)		lab_mat_sh[	(c-1)	+	(r-1)	*(blockDim.x)	] // O: scan value at North-West
#define nn_pol(c,r)		lab_mat_sh[	(c+0)	+	(r-1)	*(blockDim.x)	] // O: scan value at North
#define ne_pol(c,r)		lab_mat_sh[	(c+1)	+	(r-1)	*(blockDim.x)	] // O: scan value at North-East
#define ww_pol(c,r)		lab_mat_sh[	(c-1)	+	(r+0)	*(blockDim.x)	] // O: scan value at West
#define ee_pol(c,r)		lab_mat_sh[	(c+1)	+	(r+0)	*(blockDim.x)	] // O: scan value at West
#define sw_pol(c,r)		lab_mat_sh[	(c-1)	+	(r+1)	*(blockDim.x)	] // O: scan value at South-West
#define ss_pol(c,r)		lab_mat_sh[	(c+0)	+	(r+1)	*(blockDim.x)	] // O: scan value at South-West
#define se_pol(c,r)		lab_mat_sh[	(c+1)	+	(r+1)	*(blockDim.x)	] // O: scan value at South-West
#define cc_pol(c,r)		lab_mat_sh[	(c+0)	+	(r+0)	*(blockDim.x)	] // O: scan value at current [r,c] which is shifted by [1,1] in O

// GLOBAL VARIABLES
#define 		Vb			0	// background value
#define			Vo			1	// object value
char			buffer[255];

//---------------------------- FUNCTIONS PROTOTYPES
//		** I/O **
void read_mat(unsigned char *, unsigned int, unsigned int, char *);
void write_linmat_tiled(unsigned int *, unsigned int, unsigned int, unsigned int, unsigned int, char *);
void write_linmat_matlab(unsigned int *, unsigned int, unsigned int, unsigned int, unsigned int, char *);
//		** kernels **
//	(1)
__global__ void intra_tile_labeling( const unsigned char *,unsigned int, unsigned int * );
//	(2)
__global__ void stitching_tiles( unsigned int *,const unsigned int,const unsigned int );
//	(3)
__global__ void root_equivalence( unsigned int *,const unsigned int,const unsigned int );
//	(4)
__global__ void intra_tile_re_label(unsigned int,unsigned int *);

//		** OLD kernels **
__global__ void inter_tile_labeling( unsigned int *, bool );
//---------------------------- FUNCTIONS PROTOTYPES

void read_mat(unsigned char *urban, unsigned int nrows, unsigned int ncols, char *filename)
{
	unsigned int rr,cc;
	FILE *fid ;
	int a;
	fid = fopen(filename,"rt");
	if (fid == NULL) { printf("Error opening file!\n"); exit(1); }
	for(rr=0;rr<nrows;rr++) for(cc=0;cc<ncols;cc++) urban[cc+rr*ncols] = 0;
	for(rr=1;rr<nrows-1;rr++) for(cc=1;cc<ncols-1;cc++) { fscanf(fid, "%d",&a);	urban[cc+rr*ncols]=(unsigned char)a; }
	fclose(fid);
}

void write_linmat_tiled(unsigned int *lab_mat, unsigned int nr, unsigned int nc, unsigned int ntilesX, unsigned int ntilesY, char *filename)
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

void write_linmat_matlab(unsigned int *lab_mat, unsigned int nr, unsigned int nc, unsigned int ntilesX, unsigned int ntilesY, char *filename)
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

__global__ void inter_tile_labeling( unsigned int *lm, bool *gfound )
{
	/*
	 * IMPORTANT NOTE:
	 * 	> 	We only need to read the shared border vector from {nn,ww,ee,ss} tiles and not the entire adjacent tiles!
	 * 		Hence we should modify the code accordingly!
	 * 	>	We should allocate using cudaMallocPitch and not cudaMalloc: nvidia says that it is faster!!
	 * 		See CUDA_Runtime_API.pdf, page 92.
	 * 	>	Invertire i due if nei blocchi << if(XX_tile>=0) >> in modo che il secondo if
	 * 		(ossia accendere i thread in cc con ID uguale a quello in indice ii del bordo)
	 * 		venga eseguito solo se il primo if
	 * 		(ossia che il valore sul bordo della tile {nn,ww,ee,ss} adiacente è più piccolo)
	 * 		è verificato!
	 * 	>
	 */

	extern __shared__ unsigned int  lm_sh[];

	// http://stackoverflow.com/questions/12505750/how-can-a-global-function-return-a-value-or-break-out-like-c-c-does
	__shared__ bool someoneFoundIt;

	unsigned int r 		= threadIdx.y;
	unsigned int c 		= threadIdx.x;
	unsigned int bdx	= blockDim.x;
	unsigned int bdy	= blockDim.y;
	unsigned int otid	= bdx * r + c;

	unsigned int cc_0	= bdx*bdy*0;
	unsigned int nn_0	= bdx*bdy*1;
	unsigned int ww_0	= bdx*bdy*2;
	unsigned int ee_0	= bdx*bdy*3;
	unsigned int ss_0	= bdx*bdy*4;
	unsigned int ccR_0	= bdx*bdy*5;

	// CC tile
	int iTile			= gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int cc_tid	=	(r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile		blockDim.x*(blockDim.y-1)+threadIdx.x
							(blockDim.x - 0) * (iTile % gridDim.x)		+					// itile in orizzontale		0
							(iTile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;	// itile in verticale		0

	// NN tile
	int nn_tile			= (iTile < gridDim.x)?-1:(iTile - gridDim.x);						// tile index of nn
	int nn_tid			=	(r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile
							(blockDim.x - 0) * (nn_tile % gridDim.x)	+					// itile in orizzontale
							(nn_tile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale   192

	// WW tile
	// see in MATLAB ==> reshape(mod(0:gridDim.x*gridDim.y-1,blockDim.x),blockDim.x,blockDim.y)'
	int ww_tile			= ((iTile % gridDim.x)==0)?-1:iTile-1;								// tile index of ww			9-1=8
	int ww_tid			=	(r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile		r*10*5 +c ==> tile ad ovest della 9
							(blockDim.x - 0) * (ww_tile % gridDim.x)	+					// itile in orizzontale		5*mod(8,10)=40
							(ww_tile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale		int(8/10)*5*10*5=0

	// SS tile
	int ss_tile			= (iTile >= gridDim.x*(gridDim.y-1))?-1:(iTile + gridDim.x);		// tile index of ss
	int ss_tid			=	(r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile
							(blockDim.x - 0) * (ss_tile % gridDim.x)	+					// itile in orizzontale
							(ss_tile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale

	// EE tile
	int ee_tile			= ((iTile % gridDim.x)==gridDim.x-1)?-1:iTile+1;					// tile index of ee
	int ee_tid			=	(r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile
							(blockDim.x - 0) * (ee_tile % gridDim.x)	+					// itile in orizzontale
							(ee_tile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale

	int ii=NULL;
	*gfound = false;
	if( iTile < gridDim.x*gridDim.y )
	{
		lm_sh[cc_0+otid] 					= lm[cc_tid]; __syncthreads();
		if( ww_tile>=0 ){ lm_sh[ww_0+otid] 	= lm[ww_tid]; __syncthreads(); }
		if( nn_tile>=0 ){ lm_sh[nn_0+otid] 	= lm[nn_tid]; __syncthreads(); }
		if( ee_tile>=0 ){ lm_sh[ee_0+otid] 	= lm[ee_tid]; __syncthreads(); }
		if( ss_tile>=0 ){ lm_sh[ss_0+otid] 	= lm[ss_tid]; __syncthreads(); }
		lm_sh[ccR_0+otid] 					= lm[cc_tid]; __syncthreads();

		someoneFoundIt = true;
		while(someoneFoundIt)
		{
			someoneFoundIt = false;

			// (1) objects_stitching_nn()
			if( nn_tile>=0 )
			{
				for(ii=0;ii<blockDim.x;ii++)
				{
					if( lm_sh[ccR_0+otid]==lm_sh[ccR_0+ii] )
					{
						if( lm_sh[nn_0+blockDim.x*(blockDim.y-1)+ii] < lm_sh[ccR_0+ii] )
						{
							lm_sh[cc_0+otid] = lm_sh[nn_0+blockDim.x*(blockDim.y-1)+ii];
							someoneFoundIt = true;
							__syncthreads();
						}
					}
				}
				//lm_sh[ccR_0+otid] = lm_sh[cc_0+otid];
				//__syncthreads();
			}

			// (2) objects_stitching_ww
			if( ww_tile>=0 )
			{
				for(ii=0;ii<blockDim.y;ii++)
				{
					if( lm_sh[ccR_0+otid]==lm_sh[ccR_0+blockDim.x*ii] )
					{
						if(lm_sh[ww_0+blockDim.x*(ii+1)-1] < lm_sh[ccR_0+blockDim.x*ii])
						{
							lm_sh[cc_0+otid] = lm_sh[ww_0+blockDim.x*(ii+1)-1];
							someoneFoundIt = true;
							__syncthreads();
						}
					}
				}
				//lm_sh[ccR_0+otid] = lm_sh[cc_0+otid];
				//__syncthreads();
			}

			// (3) objects_stitching_ee
			if( ee_tile>=0 )
			{
				for(ii=0;ii<blockDim.y;ii++)
				{
					if( lm_sh[ccR_0+otid]==lm_sh[ccR_0+blockDim.x*(ii+1)-1] )
					{
						if(lm_sh[ee_0+blockDim.x*ii] < lm_sh[ccR_0+blockDim.x*(ii+1)-1])
						{
							lm_sh[cc_0+otid] = lm_sh[ee_0+blockDim.x*ii];
							someoneFoundIt = true;
							__syncthreads();
						}
					}
				}
				//lm_sh[ccR_0+otid] = lm_sh[cc_0+otid];
				//__syncthreads();
			}

			// (4) objects_stitching_ss()
			if( ss_tile>=0 )
			{
				for(ii=0;ii<blockDim.x;ii++)
				{
					if( lm_sh[ccR_0+otid]==lm_sh[ccR_0+blockDim.x*(blockDim.y-1)+ii] )
					{
						if( lm_sh[ss_0+ii] < lm_sh[ccR_0+blockDim.x*(blockDim.y-1)+ii] )
						{
							lm_sh[cc_0+otid] = lm_sh[ss_0+ii];
							someoneFoundIt = true;
							__syncthreads();
						}
					}
				}
				//lm_sh[ccR_0+otid] = lm_sh[cc_0+otid];
				//__syncthreads();
			}
			//__syncthreads();
			if(someoneFoundIt) *gfound = true;
			lm_sh[ccR_0+otid] = lm_sh[cc_0+otid];
			__syncthreads();
		}//while :: single cc
	}//blocks	 :: all cc's

	// I write the borders of all tiles synchronous without knowing
	// which tile has the lowest ID in any border and any pixel because
	// I preserve whole tile info with all duplicated borders.
	lm[cc_tid] = lm_sh[cc_0+otid];
	__syncthreads();

}//kernel

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
	unsigned int iTile		= gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int otid		= blockDim.x * r + c;
	unsigned int itid		= (r * NC + c) 								+					// dentro la 1° tile		0
							  (blockDim.x - 1) * (iTile % gridDim.x)	+					// itile in orizzontale		0
							  (iTile / gridDim.x) * (blockDim.y-1) * NC;					// itile in verticale		0
	unsigned int ttid 		= iTile*blockDim.x*blockDim.y+otid;
	unsigned int stid		= (r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile		0
			  	  	  	  	  (blockDim.x - 0) * (iTile % gridDim.x)	+					// itile in orizzontale		0
			  	  	  	  	  (iTile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale		42

	if (iTile<gridDim.x*gridDim.y)
	{
		lab_mat_sh[otid] 	= 0;
		// if (r,c) is object pixel
		if  (urban[itid]==Vo)  lab_mat_sh[otid] = ttid;
		__syncthreads();

		found = true;
		while(found)
		{
			found = false;
			if(	c>0 && r>0 && nw_pol(c,r)!=0 && nw_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = nw_pol(c,r); found = true; }
			if( r>0 && nn_pol(c,r)!=0 && nn_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = nn_pol(c,r); found = true; }
			if( c<bdx-1 && r>0 && ne_pol(c,r)!=0 && ne_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ne_pol(c,r); found = true; }
			if( c>0 && ww_pol(c,r)!=0 && ww_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ww_pol(c,r); found = true; }
			if( c<bdx-1 && ee_pol(c,r)!=0 && ee_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ee_pol(c,r); found = true; }
			if( c>0 && r<bdy-1 && sw_pol(c,r)!=0 && sw_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = sw_pol(c,r); found = true; }
			if( r<bdy-1 && ss_pol(c,r)!=0 && ss_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = ss_pol(c,r); found = true; }
			if( c<bdx-1 && r<bdy-1 && se_pol(c,r)!=0 && se_pol(c,r)<cc_pol(c,r))
				{ cc_pol(c,r) = se_pol(c,r); found = true; }
			__syncthreads();
		}

		/*
		 * 	To linearize I write using ttid.
		 * 	To leave same matrix configuration as input urban use stid instead!!
		 */
		lab_mat[ttid] = lab_mat_sh[otid];
		__syncthreads();
	}
}

__global__ void intra_tile_labeling_opt(const unsigned char *urban,unsigned int NC,unsigned int *lab_mat)
{
	extern __shared__ unsigned int  lab_mat_sh[];
	__shared__ bool found[1];

	unsigned char urban_loc;
	unsigned char neigh_loc[8];
	//unsigned int fill_val 	= 0xFFFFFFFFFFFFFFFF;
	unsigned int newLabel;
	unsigned int oldLabel;

	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int iTile		= gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int otid		= blockDim.x * r + c;
	unsigned int itid		= (r * NC + c) 								+					// dentro la 1° tile		0
							  (blockDim.x - 1) * (iTile % gridDim.x)	+					// itile in orizzontale		0
							  (iTile / gridDim.x) * (blockDim.y-1) * NC;					// itile in verticale		0
	unsigned int ttid 		= iTile*blockDim.x*blockDim.y+otid;
/*	unsigned int stid		= (r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile		0
			  	  	  	  	  (blockDim.x - 0) * (iTile % gridDim.x)	+					// itile in orizzontale		0
			  	  	  	  	  (iTile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale		42
*/
	unsigned int ex_tid		= c+1 + (r+1)*(bdx+2);

	if (iTile<gridDim.x*gridDim.y)
	{
/*
		// initialize with maximum value:
		lab_mat_sh[ex_tid] 						= fill_val;
		// **load all zeros in boundaries:
		if(c==0 && r==0){
			lab_mat_sh[0] 						= fill_val;
			lab_mat_sh[bdx+1] 					= fill_val;
			lab_mat_sh[(bdx+2)*(bdy+1)] 		= fill_val;
			lab_mat_sh[(bdx+2)*(bdy+2) -1] 		= fill_val;
		}
		if(c<bdx)
		{
			lab_mat_sh[c+1] 					= fill_val;
			lab_mat_sh[(bdx+2)*(bdy+1)+1 +c] 	= fill_val;
		}
		if(r<bdy)
		{
			lab_mat_sh[r+1] 					= fill_val;
			lab_mat_sh[(bdx+2)*(r+1)+bdx+1] 	= fill_val;
		}
		//****
*/
/*
		// fill with fill_value
		lab_mat_sh[ ex_tid - (bdx+2) 	-1 ] = fill_val;
		lab_mat_sh[ ex_tid - (bdx+2) 	+0 ] = fill_val;
		lab_mat_sh[ ex_tid - (bdx+2) 	+1 ] = fill_val;
		lab_mat_sh[ ex_tid 			-1 ]	 = fill_val;
		lab_mat_sh[ ex_tid			+1 ]	 = fill_val;
		lab_mat_sh[ ex_tid + (bdx+2) 	-1 ] = fill_val;
		lab_mat_sh[ ex_tid + (bdx+2) 	+0 ] = fill_val;
		lab_mat_sh[ ex_tid + (bdx+2) 	+1 ] = fill_val;
		__syncthreads();
*/
		// use per-thread memory facility:
		urban_loc 	 			= urban[itid];//(unsigned char)lab_mat_sh;
		// load binary objects:
		lab_mat_sh[ex_tid] 		= urban_loc; /*if( urban_loc!=Vb )*/
		__syncthreads();
		neigh_loc[0] 			= lab_mat_sh[ ex_tid - (bdx+2) 	-1 ];
		neigh_loc[1] 			= lab_mat_sh[ ex_tid - (bdx+2) 	+0 ];
		neigh_loc[2] 			= lab_mat_sh[ ex_tid - (bdx+2) 	+1 ];
		neigh_loc[3] 			= lab_mat_sh[ ex_tid 			-1 ];
		neigh_loc[4] 			= lab_mat_sh[ ex_tid			+1 ];
		neigh_loc[5] 			= lab_mat_sh[ ex_tid + (bdx+2) 	-1 ];
		neigh_loc[6] 			= lab_mat_sh[ ex_tid + (bdx+2) 	+0 ];
		neigh_loc[7] 			= lab_mat_sh[ ex_tid + (bdx+2) 	+1 ];

		// load global unique index:
		newLabel 				= ttid;
		lab_mat_sh[ex_tid] = newLabel; /*if( urban_loc!=Vb )*/
		while( 1 ){
			found[0] 			= false;
			oldLabel 			= newLabel;
			__syncthreads();

			if(urban_loc != Vb){
				if(neigh_loc[0]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid - (bdx+2) -1 ]);
				if(neigh_loc[1]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid - (bdx+2) +0 ]);
				if(neigh_loc[2]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid - (bdx+2) +1 ]);
				if(neigh_loc[3]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid 		  -1 ]);
				if(neigh_loc[4]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid			  +1 ]);
				if(neigh_loc[5]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid + (bdx+2) -1 ]);
				if(neigh_loc[6]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid + (bdx+2) +0 ]);
				if(neigh_loc[7]==urban_loc)	newLabel = min(newLabel, lab_mat_sh[ ex_tid + (bdx+2) +1 ]);
/*				newLabel = min(newLabel, lab_mat_sh[ ex_tid - (bdx+2) -1 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid - (bdx+2) +0 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid - (bdx+2) +1 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid 		  -1 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid			  +1 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid + (bdx+2) -1 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid + (bdx+2) +0 ]);
				newLabel = min(newLabel, lab_mat_sh[ ex_tid + (bdx+2) +1 ]);
*/			}	__syncthreads();

			if(oldLabel > newLabel) {
				atomicMin(&lab_mat_sh[ex_tid], newLabel); // if it is slow ==> write directly!
				//lab_mat_sh[ex_tid] = newLabel;
				//set the flag to 1 -> it is necessary to perform another iteration of the CCL solver
				found[0] 		= true;
			}	__syncthreads();
			//if no equivalence was updated, the local solution is complete
			if(found[0] == false) break;
		}

		/*  To linearize write using ttid.
		 * 	To leave same matrix configuration as input urban use stid instead!!
		 */
		if( urban_loc!=Vb ) lab_mat[ttid] = lab_mat_sh[ex_tid];
		__syncthreads();
	}
}

template <unsigned int NTHREADSX>
__global__ void stitching_tiles(	unsigned int *lab_mat,
									const unsigned int tiledimX,
									const unsigned int tiledimY		)
{	// THREADS:
	unsigned int c 		= threadIdx.x;

	// TILES:
	int nTiles			= gridDim.x * gridDim.y;
	int iTile			= gridDim.x * blockIdx.y + blockIdx.x;								// cc tile
	int nn_tile			= (iTile < gridDim.x)?-1:(iTile - gridDim.x);						// nn tile of cc tile
	int ww_tile			= ((iTile % gridDim.x)==0)?-1:iTile-1;								// ww tile of cc tile

	// SIDES:
	int c_nn_tid		=	c 											+					// (2) within-tile
							tiledimX*tiledimY * iTile;										// (1) horizontal offset
	int nn_tid			= 	c 											+					// (2) within-tile
							tiledimX*(tiledimY-1) + tiledimX*tiledimY * nn_tile;			// (1) horizontal offset
	int c_ww_tid		=	c*tiledimX 									+					// (2) within-tile
							tiledimX*tiledimY * iTile;										// (1) horizontal offset
	int ww_tid			= 	(c+1)*tiledimX-1 							+					// (2) within-tile
							tiledimX*tiledimY * ww_tile;									// (1) horizontal offset

	// SHARED:
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
			__old[ c ] = atomicMin( &lab_mat[ cc_nn[c] ], nn_ss[ c ] );
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
{	// THREADS:
	unsigned int c 		= threadIdx.x;

	// TILES:
	int nTiles			= gridDim.x * gridDim.y;
	int iTile			= gridDim.x * blockIdx.y + blockIdx.x;								// cc tile
	int nn_tile			= (iTile < gridDim.x)?-1:(iTile - gridDim.x);						// nn tile of cc tile
	int ww_tile			= ((iTile % gridDim.x)==0)?-1:iTile-1;								// ww tile of cc tile
	int ss_tile			= (iTile >= gridDim.x*(gridDim.y-1))?-1:(iTile + gridDim.x);		// tile index of ss
	int ee_tile			= ((iTile % gridDim.x)==gridDim.x-1)?-1:iTile+1;					// tile index of ee

	// SIDES:
	int c_nn_tid		=	c 											+					// (2) within-tile
							tiledimX*tiledimY * iTile;										// (1) horizontal offset
	int nn_tid			= 	c + tiledimX*(tiledimY-1)					+					// (2) within-tile
							tiledimX*tiledimY * nn_tile;									// (1) horizontal offset
	int c_ww_tid		=	c*tiledimX 									+					// (2) within-tile
							tiledimX*tiledimY * iTile;										// (1) horizontal offset
	int ww_tid			= 	(c+1)*tiledimX-1 							+					// (2) within-tile
							tiledimX*tiledimY * ww_tile;									// (1) horizontal offset

	int c_ss_tid		=	c + tiledimX*(tiledimY-1)					+					// (2) within-tile
							tiledimX*tiledimY * iTile;										// (1) horizontal offset
	int ss_tid			= 	c											+					// (2) within-tile
							tiledimX*tiledimY * ss_tile;									// (1) horizontal offset
	int c_ee_tid		=	(c+1)*tiledimX-1							+					// (2) within-tile
							tiledimX*tiledimY * iTile;										// (1) horizontal offset
	int ee_tid			= 	c*tiledimX 									+					// (2) within-tile
							tiledimX*tiledimY * ee_tile;									// (1) horizontal offset

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
			nn_ss[ c ] 		= lab_mat[ nn_tid ];
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
	unsigned int iTile		= gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int otid		= blockDim.x * r + c;
	unsigned int itid		= (r * NC + c) 								+					// dentro la 1° tile		0
							  (blockDim.x - 1) * (iTile % gridDim.x)	+					// itile in orizzontale		0
							  (iTile / gridDim.x) * (blockDim.y-1) * NC;					// itile in verticale		0
	unsigned int ttid 		= iTile*blockDim.x*blockDim.y+otid;
	unsigned int stid		= (r * gridDim.x * blockDim.x + c) 			+					// dentro la 1° tile		0
			  	  	  	  	  (blockDim.x - 0) * (iTile % gridDim.x)	+					// itile in orizzontale		0
			  	  	  	  	  (iTile / gridDim.x) * (blockDim.y-0) * gridDim.x * blockDim.x;// itile in verticale		42

	if (iTile<gridDim.x*gridDim.y)
	{
		if  (lab_mat[ttid]!=0)  lab_mat[ttid]=lab_mat[lab_mat[ttid]];
		//if  (urban[itid]==Vo)  urban[itid]=lab_mat[lab_mat[ttid]];
	}
}

int main(int argc, char **argv)
{
	// INPUTS
	unsigned int tiledimX 	= atoi( argv[1] );
	unsigned int tiledimY 	= atoi( argv[2] );
	unsigned int NC1 		= atoi( argv[3] );	// passed by JAI
	unsigned int NR1 		= atoi( argv[4] );	// passed by JAI
	const unsigned int NTHREADSX = 30;			// how to let it be variable.
	printf("NTHREADSX:\t%d\n\n",NTHREADSX);

	// MANIPULATION:
	// X dir
	unsigned int ntilesX 	= ceil( (double)(NC1+2-1) / (double)(tiledimX-1)  );
	unsigned int NC 		= ntilesX*(tiledimX-1) +1;
	// Y dir
	unsigned int ntilesY	= ceil( (double)(NR1+2-1) / (double)(tiledimY-1)  );
	unsigned int NR 		= ntilesY*(tiledimY-1) +1;

	// DECLARATIONS:
	//	Error code to check return values for CUDA calls
	cudaError_t cudaError 	= cudaSuccess;

	// size of arrays
	size_t sizeChar  		= NC*NR * sizeof(unsigned char);
	size_t sizeUintL 		= ntilesX*ntilesY*tiledimX*tiledimY * sizeof(unsigned int);
	// clocks:
	clock_t start_t, end_t;

	//	urban_cpu
	unsigned char *urban_cpu;
	cudaMallocHost(&urban_cpu,sizeChar);
	read_mat(urban_cpu, NR, NC, "/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/ALL.txt");

	start_t = clock();
/*
	cudaStream_t stream[2];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
*/
	//	urban_gpu -- stream[0]
	unsigned char *urban_gpu;
	cudaError = cudaMalloc( (void **)&urban_gpu, sizeChar );
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to allocate device array urban_gpu (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }
	//cudaError = cudaMemsetAsync( urban_gpu,0, sizeChar, stream[0] );
	cudaError = cudaMemset( urban_gpu,0, sizeChar );
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to set ZEROS in urban_gpu array on device (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }
	//cudaError = cudaMemcpyAsync( urban_gpu,urban_cpu,	sizeChar,cudaMemcpyHostToDevice, stream[0] );
	cudaError = cudaMemcpy( urban_gpu,urban_cpu,	sizeChar,cudaMemcpyHostToDevice );
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to copy array urban_cpu from host to device urban_gpu (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }

	//	lab_mat_gpu  -- stream[1]
	unsigned int  *lab_mat_gpu;
	cudaError = cudaMalloc( (void **)&lab_mat_gpu, sizeUintL );
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to allocate device array lab_mat_gpu (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }
	//cudaError = cudaMemsetAsync( lab_mat_gpu,0, sizeUintL, stream[0] );
	cudaError = cudaMemset( lab_mat_gpu,0, sizeUintL );
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to set ZEROS in lab_mat_gpu array on device (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }

	//	lab_mat_cpu
	unsigned int  *lab_mat_cpu;
	cudaMallocHost(&lab_mat_cpu,sizeUintL);

	// KERNEL INVOCATION: intra-tile labeling
	dim3 block(tiledimX,tiledimY);
	dim3 grid(ntilesX,ntilesY);
	int sh_mem 	= (tiledimX*tiledimY)*(sizeof(unsigned int)); // +sizeof(unsigned char)
	intra_tile_labeling<<<grid,block,sh_mem>>>(urban_gpu,NC,lab_mat_gpu);
	int sh_mem_2= ((tiledimX+2)*(tiledimY+2))*(sizeof(unsigned int)); // +sizeof(unsigned char)
//	intra_tile_labeling_opt<<<grid,block,sh_mem_2>>>(urban_gpu,NC,lab_mat_gpu);
	cudaError_t cudaLastErr = cudaGetLastError(); //cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost)
	if (cudaLastErr != cudaSuccess){ printf ("ERROR {intra_tile_labeling_opt} -- %s\n", cudaGetErrorString(cudaLastErr)); exit(EXIT_FAILURE); }
	//checkCudaErrors( cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost) );
/*	cudaError = cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to allocate copy array lab_mat_gpu from device to host (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }
	sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/lab_mat_cpu-intra_tile_labeling.txt");
	write_linmat_tiled(lab_mat_cpu, tiledimY, tiledimX, ntilesX, ntilesY, buffer);
*/
	dim3 block_2(tiledimX,1,1);
	dim3 grid_2(ntilesX,ntilesY,1);
	stitching_tiles<NTHREADSX><<<grid_2,block_2>>>(lab_mat_gpu,tiledimX,tiledimY);
	root_equivalence<NTHREADSX><<<grid_2,block_2>>>(lab_mat_gpu,tiledimX,tiledimY);
	// SAVE
/*	cudaError = cudaMemcpy(lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to allocate copy array lab_mat_gpu from device to host (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }
	sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/lab_mat_cpu-root_equivalence.txt");
	write_linmat_tiled(lab_mat_cpu, tiledimY, tiledimX, ntilesX, ntilesY, buffer);
*/

	// final labeling:
	intra_tile_re_label<<<grid,block,sh_mem>>>(NC,lab_mat_gpu);

	// lab_mat_cpu:
	//cudaError = cudaMemcpyAsync( lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost, stream[0] );
	cudaError = cudaMemcpy( lab_mat_cpu,lab_mat_gpu,	sizeUintL,cudaMemcpyDeviceToHost );
	if (cudaError != cudaSuccess){ fprintf(stderr, "Failed to copy array lab_mat_gpu from device to host lab_mat_cpu (error code %s)!\n", cudaGetErrorString(cudaError)); exit(EXIT_FAILURE); }

	end_t = clock();
	printf("Total time: %f [msec]\n", (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000 );

	// SAVE lab_mat to file and compare with MatLab
	sprintf(buffer,"/home/giuliano/work/Projects/LIFE_Project/LUC_gpgpu/soil_sealing/data/CUDA-code.txt");
	write_linmat_matlab(lab_mat_cpu, tiledimX, tiledimY, ntilesX, ntilesY, buffer);

	// FREE MEMORY:
	cudaFreeHost(lab_mat_cpu);
	cudaFreeHost(urban_cpu);
	cudaFree(lab_mat_gpu);
	cudaFree(urban_gpu);
/*	cudaStreamDestroy( stream[0] );
	cudaStreamDestroy( stream[1] );
*/
	// RETURN:
	return 0;
}
