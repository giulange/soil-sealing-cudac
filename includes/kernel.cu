
#include "histo.cu"

// 2D float texture
texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> tex_urban;
cudaArray* cuArrayUrban = 0;

// elimino la duplicazione dei bordi delle tile 
// (pongo a zero perché poi il valore è usato in somma)
__global__ void kCopy(unsigned int *lab_mat_out, unsigned int *lab_mat_in)
{	
	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int iTile		= gdx * biy + bix;
	unsigned int n = bdx*bdy; // pixels per tile
	unsigned int k = (r*bdx + c) + iTile*n;
	unsigned int v = lab_mat_in[k];
	if(threadIdx.x==blockDim.x-1) v=0; // questo bordo è elaborato dalla tile a destra
	if(threadIdx.y==blockDim.y-1) v=0; // questo bordo è elaborato dalla tile a sud
	lab_mat_out[k] = v;
}

// conta i pixel diversi in un intorno quadrato di lato R
__global__ void kCount(unsigned int *pixel_count, int R, int NR, int NC)
{
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	if(ix>=NC) return;
	if(iy>=NR) return;
	unsigned int n = 0;

	unsigned char val00 = tex2D (tex_urban, ix, iy);
	for(int dy=-R; dy<=R; ++dy)
	for(int dx=-R; dx<=R; ++dx)
	{
		if(dx==0 && dy==0) continue;
		if(ix+dx<0 || ix+dx>=NC) continue;
		if(iy+dy<0 || iy+dy>=NR) continue;
		if(val00 != tex2D (tex_urban, ix+dx, iy+dy)) n++;
	}
	
	pixel_count[ix + iy*NC] = n;
}

void freeTexture()
{
	cudaUnbindTexture(tex_urban);
	cudaFreeArray(cuArrayUrban);
}
int initTexture(unsigned char *h_urban, int NR, int NC)
{
	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc =
	cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	if(cuArrayUrban) cudaFreeArray(cuArrayUrban);
	cudaMallocArray(&cuArrayUrban, &channelDesc, NC, NR);
	cudaMemcpyToArray(cuArrayUrban, 0, 0, h_urban, NR*NC*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	// Set texture reference parameters
	tex_urban.addressMode[0] = cudaAddressModeClamp;
	tex_urban.addressMode[1] = cudaAddressModeClamp;
	tex_urban.filterMode = cudaFilterModePoint;
	tex_urban.normalized = false;
	
	// Bind the array to the texture reference
	cudaBindTextureToArray(tex_urban, cuArrayUrban, channelDesc);
	return 0;
}

// pixel posto al valore della label se di perimetro, 0 altrimenti
// lab_mat è organizzata per tile, ogni tile ripete il bordo delle contigue
__global__ void kPerimetro(unsigned int *perimetri, unsigned int *lab_mat, int N)
{
	unsigned int r 			= threadIdx.y;
	unsigned int c 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bdy		= blockDim.y;
	unsigned int bix		= blockIdx.x;
	unsigned int biy		= blockIdx.y;
	unsigned int gdx		= gridDim.x;
	unsigned int gdy		= gridDim.y;
	unsigned int iTile		= gdx * biy + bix;
	if(bdx!=gdx-1 && c==bdx-1) return; // questo bordo è elaborato dalla tile a destra
	if(bdy!=gdy-1 && r==bdy-1) return; // questo bordo è elaborato dalla tile a sud
	if(iTile>=gdx*gdy) return;

	unsigned int n 			= bdx*bdy; // pixels per tile
	unsigned int k 			= (r*bdx + c) + iTile*n;
	unsigned int perim 		= 0;
	unsigned int val00 		= lab_mat[k];
	
	if     (bdy==0 && r==0) perim=val00;
	else if(bdy==gdy-1 && r==bdy-1) perim=val00;
	else if(bdx==0 && c==0) perim=val00;
	else if(bdx==gdx-1 && c==bdx-1) perim=val00;
	else 
	{
		int kN = (r==0)     ? k - gdx*bdx : k - bdx;
		int kS = (r==bdy-1) ? k + gdx*bdx : k + bdx;
		int kW = (c==0)     ? k - n + (bdx-1) : k - 1;
		int kE = (c==bdx-1) ? k + n - (bdx-1) : k + 1;
		if(val00 != lab_mat[kW] ||
		   val00 != lab_mat[kE] ||
		   val00 != lab_mat[kN] ||
		   val00 != lab_mat[kS])   perim=val00; 
	}
	perimetri[k] = perim;
}

// 1 - calcolo area label (usa histo.cu)
int CalcoloArea(
	std::vector<unsigned int> &labels,
	std::vector<unsigned int> &counts,
	unsigned int *lab_mat_gpu, 
	int ntilesX, int ntilesY, int tiledimX, int tiledimY)
{ 
	std::cout << "CalcoloArea" << std::endl;
	dim3 	block(tiledimX,tiledimY,1);
	dim3 	grid(ntilesX,ntilesY,1);
	int N = ntilesX*ntilesY*tiledimX*tiledimY;
	
	thrust::device_vector<unsigned int> data(N);
	unsigned int *iArray = thrust::raw_pointer_cast( &data[0] );
	
	kCopy<<<grid,block>>>(iArray, lab_mat_gpu);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "kCopy failed\n"); exit(1);}

	std::cout << "Sparse Histogram" << std::endl;
	thrust::device_vector<unsigned int> histogram_values;
	thrust::device_vector<unsigned int> histogram_counts;
	sparse_histogram(data, histogram_values, histogram_counts);

	// copy a device_vector into an STL vector 
	int num_bins = histogram_values.size();
	labels.resize(num_bins);
	counts.resize(num_bins);
	thrust::copy(histogram_values.begin(), histogram_values.end(), labels.begin());
	thrust::copy(histogram_counts.begin(), histogram_counts.end(), counts.begin());

	std::cout << "fine CalcoloArea" << std::endl << std::endl;
	return 0;
}

// 2 - Calcolo pixel diversi dal dato in una quadrato RxR centrato sul dato 
int CalcoloFrammentazione(
	unsigned int *pixel_count, 
	unsigned char *urban_cpu, 
	int raggio,
	int NR, int NC)
{ 
	std::cout << "CalcoloFrammentazione" << std::endl;
	dim3 	block(8,8,1);
	dim3 	grid((NC+block.x-1)/block.x,(NR+block.y-1)/block.y,1);
	// attenzione: urban_cpu ha una cornice di 0
	initTexture(urban_cpu, NC, NR);

	kCount<<<grid,block>>>(pixel_count, raggio, NC, NR);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "kCount failed\n"); exit(1);}

	freeTexture();
	std::cout << "fine CalcoloFrammentazione" << std::endl << std::endl;
	return 0;
}

// 3 - perimetro (usa histo.cu)
int CalcoloPerimetro(
	std::vector<unsigned int> &labels,
	std::vector<unsigned int> &counts,
	unsigned int *lab_mat_gpu, 
	int ntilesX, int ntilesY, int tiledimX, int tiledimY)
{
	std::cout << "CalcoloPerimetro" << std::endl;
	dim3 	block(tiledimX,tiledimY,1);
	dim3 	grid(ntilesX,ntilesY,1);
	int N = ntilesX*ntilesY*tiledimX*tiledimY;
	
	thrust::device_vector<unsigned int> data(N);
	unsigned int *iArray = thrust::raw_pointer_cast( &data[0] );
	
	kPerimetro<<<grid,block>>>(iArray, lab_mat_gpu, ntilesX*ntilesY*tiledimX*tiledimY);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "kCount failed\n"); exit(1);}

	std::cout << "Sparse Histogram" << std::endl;
	thrust::device_vector<unsigned int> histogram_values;
	thrust::device_vector<unsigned int> histogram_counts;
	sparse_histogram(data, histogram_values, histogram_counts);

	// copy a device_vector into an STL vector 
	int num_bins = histogram_values.size();
	labels.resize(num_bins);
	counts.resize(num_bins);
	thrust::copy(histogram_values.begin(), histogram_values.end(), labels.begin());
	thrust::copy(histogram_counts.begin(), histogram_counts.end(), counts.begin());
	std::cout << "fine CalcoloPerimetro" << std::endl << std::endl;
	return 0;
}

