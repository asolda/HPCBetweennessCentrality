#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include "cuBCStruct.h"
#include "device_functions.h"
#include "sm_12_atomic_functions.h"
//#include "sm_20_atomic_functions.h"

#define BLOCK_SIZE 768

void initGPUGraph(const cuGraph * pCPUGraph, cuGraph *& pGPUGraph)
{
   if(pGPUGraph)
      freeGPUGraph(pGPUGraph);
   
   pGPUGraph = (cuGraph*)calloc(1, sizeof(cuGraph));
   pGPUGraph->nnode = pCPUGraph->nnode;
   pGPUGraph->nedge = pCPUGraph->nedge;

   checkCudaErrors (cudaMalloc ((void **) &(pGPUGraph->edge_node1), sizeof(int)*pGPUGraph->nedge*2));
   checkCudaErrors (cudaMalloc ((void **) &(pGPUGraph->edge_node2), sizeof(int)*pGPUGraph->nedge*2));
   checkCudaErrors (cudaMalloc ((void **) &(pGPUGraph->index_list), sizeof(int)*(pGPUGraph->nnode+1)));

   checkCudaErrors (cudaMemcpy(pGPUGraph->edge_node1, pCPUGraph->edge_node1, sizeof(int)*pGPUGraph->nedge*2, cudaMemcpyHostToDevice));
   checkCudaErrors (cudaMemcpy(pGPUGraph->edge_node2, pCPUGraph->edge_node2, sizeof(int)*pGPUGraph->nedge*2, cudaMemcpyHostToDevice));
   checkCudaErrors (cudaMemcpy(pGPUGraph->index_list, pCPUGraph->index_list, sizeof(int)*(pGPUGraph->nnode+1), cudaMemcpyHostToDevice));
}

void freeGPUGraph(cuGraph *& pGraph)
{
   if(pGraph)
   {
      checkCudaErrors (cudaFree(pGraph->edge_node1));
      checkCudaErrors (cudaFree(pGraph->edge_node2));
      checkCudaErrors (cudaFree(pGraph->index_list));
      free(pGraph);
      pGraph = NULL;
   }
}

void initGPUBC(const cuBC * pCPUBCData, cuBC *& pGPUBCData)
{
   if(pGPUBCData)
      freeGPUBC(pGPUBCData);

   pGPUBCData = (cuBC*)calloc(1, sizeof(cuBC));
   pGPUBCData->nnode = pCPUBCData->nnode;
   pGPUBCData->nedge = pCPUBCData->nedge;

   checkCudaErrors (cudaMalloc ((void **) &(pGPUBCData->numSPs), sizeof(int)*pGPUBCData->nnode));   
   checkCudaErrors (cudaMalloc ((void **) &(pGPUBCData->dependency), sizeof(float)*pGPUBCData->nnode));
   checkCudaErrors (cudaMalloc ((void **) &(pGPUBCData->distance), sizeof(int)*pGPUBCData->nnode));
   checkCudaErrors (cudaMalloc ((void **) &(pGPUBCData->nodeBC), sizeof(float)*pGPUBCData->nnode));
   checkCudaErrors (cudaMemset ((void *) pGPUBCData->nodeBC, 0, sizeof(float)*pGPUBCData->nnode));
   checkCudaErrors (cudaMalloc ((void **) &(pGPUBCData->successor), sizeof(bool)*pGPUBCData->nedge*2));
}

void freeGPUBC(cuBC *& pBCData)
{
   if(pBCData)
   {
      checkCudaErrors (cudaFree(pBCData->successor));
      checkCudaErrors (cudaFree(pBCData->numSPs));      
      checkCudaErrors (cudaFree(pBCData->distance));
      checkCudaErrors (cudaFree(pBCData->dependency));
      checkCudaErrors (cudaFree(pBCData->nodeBC));
      free(pBCData);
      pBCData = NULL;
   }
}

void clearGPUBC(cuBC * pBCData)
{
   if(pBCData)
   {
      checkCudaErrors (cudaMemset ((void *) pBCData->numSPs, 0, sizeof(int)*pBCData->nnode));
      checkCudaErrors (cudaMemset ((void *) pBCData->dependency, 0, sizeof(float)*pBCData->nnode));
      checkCudaErrors (cudaMemset ((void *) pBCData->distance, 0xff, sizeof(int)*pBCData->nnode));
      checkCudaErrors (cudaMemset ((void *) pBCData->successor, 0, sizeof(bool)*pBCData->nedge*2));
   }
}

void copyBackGPUBC(const cuBC * pGPUBCData, const cuBC * pCPUBCData)
{
   if(pCPUBCData && pGPUBCData)
   {
      checkCudaErrors (cudaMemcpy(pCPUBCData->nodeBC, pGPUBCData->nodeBC, sizeof(float)*pGPUBCData->nnode, cudaMemcpyDeviceToHost));
   }
}


void copyBCData2GPU(const cuBC * pGPUBCData, const cuBC * pCPUBCData)
{
   checkCudaErrors (cudaMemcpy(pGPUBCData->successor, pCPUBCData->successor, sizeof(bool)*pGPUBCData->nedge*2, cudaMemcpyHostToDevice));
   checkCudaErrors (cudaMemcpy(pGPUBCData->numSPs, pCPUBCData->numSPs, sizeof(int)*pGPUBCData->nnode, cudaMemcpyHostToDevice));
   checkCudaErrors (cudaMemcpy(pGPUBCData->distance, pCPUBCData->distance, sizeof(int)*pGPUBCData->nnode, cudaMemcpyHostToDevice));
   checkCudaErrors (cudaMemcpy(pGPUBCData->dependency, pCPUBCData->dependency, sizeof(float)*pGPUBCData->nnode, cudaMemcpyHostToDevice));
   checkCudaErrors (cudaMemcpy(pGPUBCData->nodeBC, pCPUBCData->nodeBC, sizeof(float)*pGPUBCData->nnode, cudaMemcpyHostToDevice));
}

void copyBCData2CPU(const cuBC * pCPUBCData, const cuBC * pGPUBCData)
{
   checkCudaErrors (cudaMemcpy(pCPUBCData->successor, pGPUBCData->successor, sizeof(bool)*pGPUBCData->nedge*2, cudaMemcpyDeviceToHost));
   checkCudaErrors (cudaMemcpy(pCPUBCData->numSPs, pGPUBCData->numSPs, sizeof(int)*pGPUBCData->nnode, cudaMemcpyDeviceToHost));
   checkCudaErrors (cudaMemcpy(pCPUBCData->distance, pGPUBCData->distance, sizeof(int)*pGPUBCData->nnode, cudaMemcpyDeviceToHost));
   checkCudaErrors (cudaMemcpy(pCPUBCData->dependency, pGPUBCData->dependency, sizeof(float)*pGPUBCData->nnode, cudaMemcpyDeviceToHost));   
   checkCudaErrors (cudaMemcpy(pCPUBCData->nodeBC, pGPUBCData->nodeBC, sizeof(float)*pGPUBCData->nnode, cudaMemcpyDeviceToHost));
}

struct node_list
{
   int * nodes;
   int size;
};

__global__ void cuda_computeBC_block(const cuGraph graph, 
                                     const node_list srcs)
{
   __shared__ cuBC  bcData;
   __shared__ float toprocess;
   __shared__ int   edge2;
   if(threadIdx.x==0)
   {    
      bcData = const_BCDatas[blockIdx.x];
      edge2  = (bcData.nedge<<1);
   }
   int   * pNumSPs     = const_BCDatas[blockIdx.x].numSPs;
   float * pDependency = const_BCDatas[blockIdx.x].dependency;
   __syncthreads();

   for(int src_idx = blockIdx.x; src_idx < srcs.size; src_idx += NUM_BLOCKS)
   //int src_idx = blockIdx.x;
   {
      // clear data
      for(int node_idx = threadIdx.x; node_idx < bcData.nnode; node_idx += blockDim.x)
      {
         pNumSPs[node_idx] = 0;
         pDependency[node_idx] = 0.f;
         bcData.distance[node_idx] = -1;
      }
      for(int node_idx = threadIdx.x; node_idx < edge2; node_idx += blockDim.x)
      {
         bcData.successor[node_idx] = false;
      }

      __syncthreads();

      // initial BFS
      if(threadIdx.x==0)
      {
         int node = srcs.nodes[src_idx];
         // bcData.numSPs[node] = 1;
         bcData.distance[node] = 0;
         toprocess = 1;
         int nb_cur = graph.index_list[node];
         int nb_end = graph.index_list[node+1];
         for(; nb_cur<nb_end; nb_cur++)
         {
            node = graph.edge_node2[nb_cur];
            bcData.distance[node] = 1;
            bcData.numSPs[node] = 1;
         }
      }
      
      int distance = 0;
      __syncthreads();
            
      // BFS
      while(toprocess>0)
      {         
         __syncthreads();
         toprocess = 0;
         distance ++;
         __syncthreads();

         for(int edge_idx = threadIdx.x; edge_idx < edge2; edge_idx += blockDim.x)
         {
            int from = graph.edge_node1[edge_idx];
            if(bcData.distance[from] == distance)
            {
               int to          = graph.edge_node2[edge_idx];
               int to_distance = bcData.distance[to];
               if(to_distance<0)
               {
                  bcData.distance[to] = to_distance = distance+1;
                  toprocess =1;
               }
               //else if(to_distance<distance)
               //{
               //   bcData.successor[edge_idx] = true; // predecessor
               //}
               if(to_distance>distance)
               {
                  bcData.successor[edge_idx] = true; // successor
                  atomicAdd(&pNumSPs[to], pNumSPs[from]);
               }
            }
         }
         
         __syncthreads();
      }


      // compute BC
      while(distance >1)
      {
         distance--;
         for(int node_idx = threadIdx.x; node_idx < bcData.nnode; node_idx += blockDim.x)
         {
	         if (bcData.distance[node_idx] == distance)	// n_id is in frontier
	         {
		         // get neighbor indices (starting index and number of indices)
		         int nb_cur = graph.index_list[node_idx];
		         int nb_end = graph.index_list[node_idx+1];		    

		         // for each nieghboring index...
		         float numSPs = bcData.numSPs[node_idx];
		         float dependency = 0.0f;		   
		         for (; nb_cur < nb_end; nb_cur++)
		         {			
			         if (bcData.successor[nb_cur])	// neighbor is a successor, using successor32bits
			         {
				         int nb_id = graph.edge_node2[nb_cur];
				         dependency += (1.0f + bcData.dependency[nb_id]) * numSPs / bcData.numSPs[nb_id];
			         }
		         }
		         bcData.dependency[node_idx] = dependency;
		         bcData.nodeBC[node_idx] += dependency;
	         }
         }

         __syncthreads();
      }

   }
}



__global__ void cuda_sumBC_block()
{
   __shared__ cuBC  bcData[NUM_BLOCKS];
   if(threadIdx.x<NUM_BLOCKS)
   {    
      bcData[threadIdx.x] = const_BCDatas[threadIdx.x];
   }

   __syncthreads();

   int node_idx = threadIdx.x + blockIdx.x * blockDim.x;
   if(node_idx<bcData[0].nnode)
   {
      float sum = bcData[0].nodeBC[node_idx];
	   sum += bcData[1].nodeBC[node_idx];
      sum += bcData[2].nodeBC[node_idx];
      sum += bcData[3].nodeBC[node_idx];
      sum += bcData[4].nodeBC[node_idx];
      sum += bcData[5].nodeBC[node_idx];
      sum += bcData[6].nodeBC[node_idx];
      sum += bcData[7].nodeBC[node_idx];
      sum += bcData[8].nodeBC[node_idx];
      sum += bcData[9].nodeBC[node_idx];
      sum += bcData[10].nodeBC[node_idx];
      sum += bcData[11].nodeBC[node_idx];
      sum += bcData[12].nodeBC[node_idx];
      sum += bcData[13].nodeBC[node_idx];
      sum += bcData[14].nodeBC[node_idx];
      sum += bcData[15].nodeBC[node_idx];
      sum += bcData[16].nodeBC[node_idx];
      sum += bcData[17].nodeBC[node_idx];
      sum += bcData[18].nodeBC[node_idx];
      sum += bcData[19].nodeBC[node_idx];
      sum += bcData[20].nodeBC[node_idx];
      sum += bcData[21].nodeBC[node_idx];
      sum += bcData[22].nodeBC[node_idx];
      sum += bcData[23].nodeBC[node_idx];
      sum += bcData[24].nodeBC[node_idx];
      sum += bcData[25].nodeBC[node_idx];
      sum += bcData[26].nodeBC[node_idx];
      sum += bcData[27].nodeBC[node_idx];
      sum += bcData[28].nodeBC[node_idx];
      sum += bcData[29].nodeBC[node_idx];
      bcData[0].nodeBC[node_idx] = sum * 0.5f;
   }
}

void gpuComputeBCOpt(const cuGraph * pGraph, cuBC * pBCData, int start, int end)
{
   // init bc data for each block
   cuBC pCPUBCDatas[NUM_BLOCKS];
   cuBC * pTmp[NUM_BLOCKS]; 
   pCPUBCDatas[0] = *pBCData;
   for(int i=1; i<NUM_BLOCKS; i++)
   {
      pTmp[i] = NULL;
      initGPUBC(pBCData, pTmp[i]);
      pCPUBCDatas[i] = *pTmp[i];
   }
   cudaMemcpyToSymbol(const_BCDatas, pCPUBCDatas, sizeof(cuBC)*NUM_BLOCKS);   

   // prepare source node list
   node_list srcs;
   srcs.size = pGraph->nnode;
   checkCudaErrors (cudaMalloc ((void **) &(srcs.nodes), sizeof(int)*(srcs.size)));
   int * tmp = (int*)calloc((srcs.size), sizeof(int));
   for(int i=start; i<end; i++) tmp[i] = i;
   checkCudaErrors (cudaMemcpy(srcs.nodes, tmp, sizeof(int)*(srcs.size), cudaMemcpyHostToDevice));
   free(tmp);
   
   // call kernels
   cuda_computeBC_block<<<NUM_BLOCKS, BLOCK_SIZE>>>(*pGraph, srcs);

   int num_blocks = (pBCData->nnode + BLOCK_SIZE-1)/BLOCK_SIZE;
   cuda_sumBC_block<<<num_blocks, BLOCK_SIZE>>>();

   checkCudaErrors (cudaFree(srcs.nodes));
   for(int i=1; i<NUM_BLOCKS; i++)
      freeGPUBC(pTmp[i]);
}