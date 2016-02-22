
// 1 - calcolo area label (usa histo.cu)
int CalcoloArea(
	std::vector<unsigned int> &labels,
	std::vector<unsigned int> &counts,
	unsigned int *lab_mat_gpu, 
	int ntilesX, int ntilesY, int tiledimX, int tiledimY);
// 2 - Calcolo pixel diversi dal dato in una quadrato RxR centrato sul dato 
int CalcoloFrammentazione(
	unsigned int *pixel_count, 
	unsigned char *urban_cpu, 
	int raggio,
	int NR, int NC);
// 3 - perimetro (usa histo.cu)
int CalcoloPerimetro(
	std::vector<unsigned int> &labels,
	std::vector<unsigned int> &counts,
	unsigned int *lab_mat_gpu, 
	int ntilesX, int ntilesY, int tiledimX, int tiledimY);