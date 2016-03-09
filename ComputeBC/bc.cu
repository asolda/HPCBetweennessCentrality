#include <stdio.h>
#include <string.h>
#include <string>
#include "cuBCStruct.h"
#include "constant.h"
#include "graph_indexed.h"

#include "BC_gpu.cu"
#include "BC_cpu.cpp"

/*
* argv[1] = filename
* argv[2] = {0, 1}, 0 = CPU, 1 = GPU
*/
int main(int argc, char *argv[]) {

	if(argc != 5) {
		std::cout << "Error. Usage: " << argv[0] << " graph.edge [0 (CPU), 1 (GPU)]";
	}

	char *filename = argv[1];

	//caricamento grafo da file
	GraphIndexed* pGraph = new GraphIndexed();
   	if(!pGraph->Load(filename)) {
   		std::cout << "Error while loading file" << std::endl;
      	return -1;
   	}

   	//inizializzazione CPU
   	cuGraph* pCUGraph = NULL;
   	cuBC*    pBCData  = NULL;
   	initGraph(pGraph, pCUGraph);
   	initBC(pCUGraph, pBCData);

   	//inizializzazione GPU
   	cuGraph* pGPUCUGraph = NULL;
   	cuBC*    pGPUBCData  = NULL;
    initGPUGraph(pCUGraph, pGPUCUGraph);
    initGPUBC(pBCData, pGPUBCData);

    int start = atoi(argv[3]);
    int end = atoi(argv[4]);
    if(start<0)
      start = 0;
    if(end>pCUGraph->nnode)
      end = pCUGraph->nnode;
    std::cout << "sono prima del bivio" << std::endl;
    if(strcmp(argv[2], "C") == 0) {
    	//use CPU
    	//compute BC (parametri 3 e 4: start, end)
    	cpuComputeBCOpt(pCUGraph, pBCData, start, end);
    } else {
    	//use GPU
    	//compute BC (parametri 3 e 4: start, end)
    	gpuComputeBCOpt(pGPUCUGraph, pGPUBCData, start, end);
    	//copy results
    	copyBackGPUBC(pGPUBCData, pBCData);
    	//free memory
    	freeGPUGraph(pGPUCUGraph);
      freeGPUBC(pGPUBCData);
    }
    std::cout << "sono dopo il bivio" << std::endl;
    //print all BCs
    const GraphIndexed::Nodes & nodes = pGraph->GetNodes();
   	for(int i=0; i<pBCData->nnode; i++) {
      std::cout << nodes[i] << "\t" << pBCData->nodeBC[i] << std::endl;
   	}
    //free all remaining memory 
    freeGraph(pCUGraph);
   	freeBC(pBCData);
   	delete pGraph;   

	return 0;
}