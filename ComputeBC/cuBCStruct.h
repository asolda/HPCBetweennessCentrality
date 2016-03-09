#ifndef _CU_BC_STRUCT_H_
#define _CU_BC_STRUCT_H_

#define EPSILON 0.0001f

#define MAX(x,y) (x>y?x:y)
#define MIN(x,y) (x<y?x:y)
#define ABS(x) (x>0.0f?x:-1.0f*x)
#define FLOOR(x) (float)(x>0.0f?(int)(x):(int)((x)-1.0f))
#define BIT2INT(x) ((x+31)>>5)

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "graph_indexed.h"

#define COMPUTE_EDGE_BC
#define NUM_BLOCKS        30   // for GTX 280
#define NUM_THREADS       768  // for GTX 280
#define NUM_LANES         8
#define WARP_SIZE         32

struct cuGraph
{
   int nnode;           // number of nodes
   int nedge;           // number of edges
   int * edge_node1;    // one end node of edges, edge*2
   int * edge_node2;    // the other end node of edges, edge*2
   int * index_list;    // start and length of neighbors, nnode+1
#ifdef COMPUTE_EDGE_BC
   int * edge_id;       // edge of edges, edge*2
#endif
};

struct cuBC
{
   int nnode;
   int nedge;
   int toprocess;
   int * numSPs;        // number of shortest paths, nnode
   int * distance;      // shortest path distance, nnode
   float * dependency;  // shortest path dependency, nnode
   float * nodeBC;      // BC computed on nodes, nnode
   bool * successor;    // bit mask for successor, 2*nedge bits
#ifdef COMPUTE_EDGE_BC
   float * edgeBC;      // BC computed on edges, nedge
#endif
};

// cpu functions
void initGraph(const GraphIndexed * pGraph, cuGraph *& pCUGraph);
void freeGraph(cuGraph *& pGraph);
void initBC(const cuGraph * pGraph, cuBC *& pBCData);
void freeBC(cuBC *& pBCData);
void clearBC(cuBC * pBCData);

void cpuHalfBC(cuBC * pBCData);
void cpuSaveBC(const cuBC * pBCData, const char* filename);
void cpuSaveBC(const GraphIndexed * pGraph, const cuBC * pBCData, const char* filename);
void cpuLoadBC(const cuBC * pBCData, const char* filename);

// cpu optimized version
void cpuComputeBCOpt(const cuGraph * pGraph, cuBC * pBCData);
int  cpuBFSOpt(const cuGraph * pGraph, cuBC * pBCData, int startNode, std::vector<int> & traversal);
int  cpuBFSOpt(const cuGraph * pGraph, cuBC * pBCData, int startNode, std::vector<int> & traversal, int wavefrontLmt);
void cpuUpdateBCOpt(const cuGraph * pGraph, cuBC * pBCData, int distance, const std::vector<int> & traversal);

// cpu approximate version
void cpuComputeBCOptApprox(const cuGraph * pGraph, cuBC * pBCData);

// gpu functions
void initGPUGraph(const cuGraph * pCPUGraph, cuGraph *& pGPUGraph);
void freeGPUGraph(cuGraph *& pGraph);
void initGPUBC(const cuBC * pCPUBCData, cuBC *& pGPUBCData);
void freeGPUBC(cuBC *& pBCData);
void clearGPUBC(cuBC * pBCData);
void copyBackGPUBC(const cuBC * pGPUBCData, const cuBC * pCPUBCData);

void copyBCData2GPU(const cuBC * pGPUBCData, const cuBC * pCPUBCData);
void copyBCData2CPU(const cuBC * pCPUBCData, const cuBC * pGPUBCData);

// gpu optimized version
void gpuComputeBCOpt(const cuGraph * pGraph, cuBC * pBCData);

// gpu approximate version
void gpuComputeBCApprox(const cuGraph * pGraph, cuBC * pBCData);

// measure BC approximation error
float measureBCApproxError(cuBC * pBCData, cuBC * pBCDataApprox);

#endif
