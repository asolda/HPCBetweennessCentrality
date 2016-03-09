#ifndef _CONSTANT_H_
#define _CONSTANT_H_

#include "cuBCStruct.h"

// constant memory array of BC structs
// 30 * 8 * 4 = 960 bytes
__constant__ cuBC const_BCDatas[NUM_BLOCKS];

// constant memory array of intermediate arrays
__constant__ float* const_iArray[NUM_BLOCKS];

#endif //_CONSTANT_H_
