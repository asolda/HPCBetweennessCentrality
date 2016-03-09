#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <queue>
#include <vector>
#include "cuBCStruct.h"

void initGraph(const GraphIndexed * pGraph, cuGraph *& pCUGraph)
{
   if(pCUGraph)
      freeGraph(pCUGraph);
   
   pCUGraph = (cuGraph*)calloc(1, sizeof(cuGraph));
   pCUGraph->nnode = pGraph->NumberOfNodes();
   pCUGraph->nedge = pGraph->NumberOfEdges();

   pCUGraph->edge_node1 = (int*)calloc(pCUGraph->nedge*2, sizeof(int));
   pCUGraph->edge_node2 = (int*)calloc(pCUGraph->nedge*2, sizeof(int));
   pCUGraph->index_list = (int*)calloc(pCUGraph->nnode+1, sizeof(int));

   int offset = 0;
   int* edge_node1 = pCUGraph->edge_node1;
   int* edge_node2 = pCUGraph->edge_node2;
   int* index_list = pCUGraph->index_list;

   for(int i=0; i<pGraph->NumberOfNodes(); i++)
   {
      GraphIndexed::Nodes neighbors = pGraph->GetNodes(i);
      GraphIndexed::Nodes::iterator iter1;
      for(iter1=neighbors.begin(); iter1!=neighbors.end(); iter1++)
      {
         *edge_node1++ = i;
         *edge_node2++ = (*iter1);
      }
      *index_list++ = offset;
      offset += neighbors.size();
   }
   *index_list = offset;
}

void freeGraph(cuGraph *& pGraph)
{
   if(pGraph)
   {
      free(pGraph->edge_node1);
      free(pGraph->edge_node2);
      free(pGraph->index_list);
      free(pGraph);
      pGraph = NULL;
   }
}

void initBC(const cuGraph * pGraph, cuBC *& pBCData)
{
   if(pBCData)
      freeBC(pBCData);

   pBCData = (cuBC*)calloc(1, sizeof(cuBC));
   pBCData->nnode = pGraph->nnode;
   pBCData->nedge = pGraph->nedge;
      
   pBCData->numSPs = (int*)calloc(pBCData->nnode, sizeof(int));
   pBCData->dependency = (float*)calloc(pBCData->nnode, sizeof(float));
   pBCData->distance = (int*)calloc(pBCData->nnode, sizeof(int));
   pBCData->nodeBC = (float*)calloc(pBCData->nnode, sizeof(float));
   pBCData->successor  = (bool*)calloc(pBCData->nedge*2, sizeof(bool));
}

void freeBC(cuBC *& pBCData)
{
   if(pBCData)
   {
      free(pBCData->successor);
      free(pBCData->numSPs);
      free(pBCData->nodeBC);
      free(pBCData->dependency);
      free(pBCData->distance);
      free(pBCData);
      pBCData = NULL;
   }
}

void clearBC(cuBC * pBCData)
{
   if(pBCData)
   {
      pBCData->toprocess = 0;
      memset(pBCData->numSPs, 0, pBCData->nnode*sizeof(int));
      memset(pBCData->dependency, 0, pBCData->nnode*sizeof(float));
      memset(pBCData->distance, 0xff, pBCData->nnode*sizeof(int));
      memset(pBCData->successor, 0, pBCData->nedge*2*sizeof(bool));
      // do not clear nodeBC & edgeBC which is accumulated through iterations
   }
}

void cpuHalfBC(cuBC * pBCData)
{
   for(int i=0; i<pBCData->nnode; i++)
      pBCData->nodeBC[i] *= 0.5f;
}

int  cpuBFSOpt(const cuGraph * pGraph, cuBC * pBCData, int startNode, std::vector<int> & traversal)
{
   pBCData->numSPs[startNode] = 1;
   pBCData->distance[startNode] = 0;
   pBCData->toprocess = 1;   
   int distance  = 0;
   int index = 0;
   std::deque<int> fifo;
   fifo.push_back(startNode);
   while(!fifo.empty())
   {  
      int from = fifo.front();
      fifo.pop_front();
      traversal[index++] = from;

      distance = pBCData->distance[from];     

      int nb_cur = pGraph->index_list[from];
      int nb_end = pGraph->index_list[from+1];
      for(; nb_cur<nb_end; nb_cur++)
      {
         int nb_id = pGraph->edge_node2[nb_cur];
         int nb_distance = pBCData->distance[nb_id];
                     
         if(nb_distance<0)
         {
            pBCData->distance[nb_id] = nb_distance = distance+1;
            fifo.push_back(nb_id);
         }
         else if(nb_distance<distance)
         {
            pBCData->numSPs[from] += pBCData->numSPs[nb_id];
         }
         if(nb_distance>distance)
         {
            pBCData->successor[nb_cur] = true;
         }
      }
   }
   return distance;
}

void cpuUpdateBCOpt(const cuGraph * pGraph, cuBC * pBCData, int distance, const std::vector<int> & traversal)
{  
   std::vector<int>::const_reverse_iterator criter;
   for(criter=traversal.rbegin(); criter!=traversal.rend(); criter++)
   {
      int from = (*criter);

      if(pBCData->distance[from]>=distance)
         continue;
      
      int nb_cur = pGraph->index_list[from];
      int nb_end = pGraph->index_list[from+1];
      
      float numSPs = pBCData->numSPs[from]; 
      float dependency = 0;
      for(; nb_cur<nb_end; nb_cur++)
      {            
         
         if(pBCData->successor[nb_cur])
         {
            int nb_id = pGraph->edge_node2[nb_cur];
            //std::cout << "sono in cpu cpuUpdateBCOpt 6" << std::endl;
            float partialDependency = numSPs / pBCData->numSPs[nb_id];
            partialDependency *= (1.0f + pBCData->dependency[nb_id]);
            
            dependency += partialDependency;
            //int edgeid = pGraph->edge_id[nb_cur];
            //pBCData->edgeBC[edgeid] += partialDependency;
         }
         
      }
      pBCData->dependency[from] = dependency;
      pBCData->nodeBC[from] += dependency;
   } 
}

// cpu optimized version
void cpuComputeBCOpt(const cuGraph * pGraph, cuBC * pBCData, int start, int end)
{   
   for(int i=start; i<end; i++)
   {
      clearBC(pBCData);
      std::vector<int> traversal;
      traversal.resize(pGraph->nnode);
      int distance = cpuBFSOpt(pGraph, pBCData, i, traversal);
      float bk = pBCData->nodeBC[i];
      cpuUpdateBCOpt(pGraph, pBCData, distance, traversal);
      pBCData->nodeBC[i] = bk;
   }

   cpuHalfBC(pBCData);
}
