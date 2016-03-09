#include <iostream>
#include <stack>
#include <list>
#include <queue>
#include <string>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include "mpi.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuBCStruct.h"
#include "constant.h"
#include "graph_indexed.h"

#include "BC_gpu.cu"
#include "BC_cpu.cpp"

#define DEVICE_COUNT_FAIL 8

#define MASTER 0

#define FROM_MASTER 2
#define TO_MASTER 3

int main(int argc, char **argv) {

	int i;
	int rank, size;
	//processor name
	char *p_name;
	int name_length;
    int handshakeBit;
    std::vector<int> masters;
	MPI_Status status;
	//store available GPUs for each node
	int devCount;
	int totalCount;
    int recvTime, maxTime;

    int unit;

    double begintime, endtime;

    //key: proc name, value: master && numdev
    std::map<std::string, std::pair<int, int> > fullMap;
    //key: processor name, value: process rank
    std::map<std::string, int> procMap;
    //key: processor name, value: master_id && slave_ids vector
    std::map<std::string, std::pair<int, std::vector<int> > > subMasterMap;
	
	MPI_Init(&argc, &argv);

    int CPU_VALUE = atoi(argv[2]);
    int GPU_VALUE = 10 - CPU_VALUE;

	MPI_Comm_size(MPI_COMM_WORLD, &size); 
	//size = MPI::COMM_WORLD.Get_size();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    //rank = MPI::COMM_WORLD.Get_rank();

    p_name = (char*) malloc(MPI_MAX_PROCESSOR_NAME * sizeof(char));

    if(rank != MASTER) { //SLAVE
    	MPI_Get_processor_name(p_name, &name_length);
    	
        MPI_Send(p_name, name_length, MPI_CHAR, MASTER, TO_MASTER, MPI_COMM_WORLD);

        MPI_Recv(&handshakeBit, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        if(handshakeBit == 1) {
            //I am processor's master
            cudaGetDeviceCount(&devCount);
            MPI_Send(&devCount, 1, MPI_INT, MASTER, TO_MASTER, MPI_COMM_WORLD);            
        }
    } else { //MASTER 
        begintime = MPI::Wtime();
    
        std::pair<std::string, std::pair<int, std::vector<int> > > nextEntry;
        std::pair<std::string, int> toInsert;
    	for(i = 1; i < size; i++) {
    		MPI_Recv(p_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, TO_MASTER, MPI_COMM_WORLD, &status);
            std::string str(p_name);
            //map master -> slaves
            if(subMasterMap.find(str) != subMasterMap.end()) {
                subMasterMap[str].second.push_back(i);
            } else {
                nextEntry.first = str;
                nextEntry.second.first = i;
                subMasterMap.insert(nextEntry);
            }
            toInsert.first = str;
            toInsert.second = i;
            procMap.insert(toInsert); 
    	}
        
        std::map<std::string, int>::iterator iter;
        for(iter = procMap.begin(); iter != procMap.end(); ++iter) {
            //std::cout << "Processor name: " << iter->first << " Processor master: " << iter->second << "\n";
            masters.push_back(iter->second);
        }
        for(i = 1; i < size; i++) {
            //sending 1 to masters and 0 to non-masters
            if(find(masters.begin(), masters.end(), i) != masters.end()){
                //masters found
                handshakeBit = 1;
                MPI_Send(&handshakeBit, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
            } else {
                handshakeBit = 0;
                MPI_Send(&handshakeBit, 1, MPI_INT, i, FROM_MASTER, MPI_COMM_WORLD);
            }
        }

        //receiving numbers of cuda devices
        std::pair<std::string, std::pair<int, int> > newToInsert;
        std::pair<int, int> nextValue;
        totalCount = 0;
        for(std::vector<int>::iterator it = masters.begin(); it != masters.end(); ++it) {
            MPI_Recv(&devCount, 1, MPI_INT, *it, TO_MASTER, MPI_COMM_WORLD, &status);
            if(devCount >= DEVICE_COUNT_FAIL || devCount < 0) devCount = 0;
            totalCount += devCount;
            for(iter = procMap.begin(); iter != procMap.end(); ++iter) {
                if(*it == iter->second) {
                    //std::cout << "Processor name: " << iter->first << " Processor master: " << *it << " Number of cuda devices: " << devCount << "\n";
                    newToInsert.first = iter->first;
                    nextValue.first = *it;
                    nextValue.second = devCount;
                    newToInsert.second = nextValue;
                    fullMap.insert(newToInsert);
                }
                continue;
            }
            
        }

        //computing minimum input size
        unit = (size * CPU_VALUE) + (totalCount * GPU_VALUE);

        //printing info
        /*
        std::cout << "CPU number = " << size << " CPU value = " << CPU_VALUE << "\nGPU number = " << totalCount << " GPU value = " << GPU_VALUE<< "\nUnit value: " << unit << "\n";

        std::cout << "\nMore info about cluster capacities:\n\n";
        std::map<std::string, std::pair<int, int> >::iterator info;
        for(info = fullMap.begin(); info != fullMap.end(); ++info) {        
            std::cout << "Node: " << info->first << " Node master: " << info->second.first << "\n";
            std::cout << "Available CPUs: " << size / fullMap.size() << " Available GPUs: " << info->second.second << "\n";
            std::cout << "Processes on this node: ";
            std::cout << "(" << subMasterMap[info->first].first << ") ";
            std::vector<int> slaves = subMasterMap[info->first].second;
            for(i = 0; i < slaves.size(); i++) {
                std::cout << slaves[i] << " ";
            }
            std::cout << "\n\n";
        }
        */
    }//FINE MASTER

    char *filename = argv[1];

    //caricamento grafo da file
    GraphIndexed* pGraph = new GraphIndexed();
    if(!pGraph->Load(filename)) {
        std::cout << "Error while loading file" << std::endl;
        return -1;
    }

    int n = pGraph->NumberOfNodes();

    if(rank == MASTER) {
        int slice = n / unit;
        int currentPos = 0;

        //use GPU, start, end
        int toSend[3] = {0, 0, 0};

        //work distribution
        int availGPUs = 0;
        handshakeBit = 0;
        std::map<std::string, std::pair<int, int> >::iterator info;
        //for each processor
        for(info = fullMap.begin(); info != fullMap.end(); ++info) { 
            std::vector<int> slaves = subMasterMap[info->first].second;
            availGPUs = info->second.second;
            //for each slave on proc
            for(i = 0; i < slaves.size(); i++) { 
                if(availGPUs > 0 && GPU_VALUE > 0) {
                    //use GPU
                    toSend[0] = 1;
                    toSend[1] = currentPos;
                    currentPos += ((slice) * GPU_VALUE) + 1;
                    toSend[2] = currentPos;
                    availGPUs--;
                } else {
                    toSend[0] = 0;
                    toSend[1] = currentPos;
                    currentPos += ((slice) * CPU_VALUE) + 1;
                    toSend[2] = currentPos;
                }      
                MPI_Send(toSend, 3, MPI_INT, slaves[i], FROM_MASTER, MPI_COMM_WORLD);

            }
            //send to master
            MPI_Send(toSend, 3, MPI_INT, subMasterMap[info->first].first, FROM_MASTER, MPI_COMM_WORLD);
        }
    }

    //inizializzazione CPU
    cuGraph* pCUGraph = NULL;
    cuBC*    pBCData  = NULL;
    initGraph(pGraph, pCUGraph);
    initBC(pCUGraph, pBCData);

    if(rank != MASTER) {
        int recv[3];

        MPI_Recv(recv, 3, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        if(recv[0] == 1) {
            //inizializzazione GPU
            cuGraph* pGPUCUGraph = NULL;
            cuBC*    pGPUBCData  = NULL;
            initGPUGraph(pCUGraph, pGPUCUGraph);
            initGPUBC(pBCData, pGPUBCData);

            //computing BC on GPU
            gpuComputeBCOpt(pGPUCUGraph, pGPUBCData, recv[1], recv[2]);
            //copy results
            copyBackGPUBC(pGPUBCData, pBCData);
            //free memory
            freeGPUGraph(pGPUCUGraph);
            freeGPUBC(pGPUBCData);
        } else {
            //compute BC on CPU
            cpuComputeBCOpt(pCUGraph, pBCData, recv[1], recv[2]);
        }
    }

    float *BCs = (float*)calloc(n, sizeof(float));

    MPI_Reduce(pBCData->nodeBC, BCs, n, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == MASTER) {
        endtime = MPI::Wtime();
        std::cout << "Time: " << endtime - begintime << " ms Graph: " << filename << " alpha: " << CPU_VALUE << std::endl;
    }

    freeGraph(pCUGraph);
    freeBC(pBCData);
    delete pGraph;  

    MPI_Finalize(); 
}