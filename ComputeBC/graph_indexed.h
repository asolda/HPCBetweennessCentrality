#ifndef _GRAPH_INDEXED_H_
#define _GRAPH_INDEXED_H_

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class GraphIndexed
{
public:
  typedef std::vector<int> Nodes;
  typedef std::vector<int> NodeEdge;
  typedef std::map<int, int> NodeIndex;
  typedef std::vector<NodeEdge> NodeEdges;
  typedef std::vector<std::pair<int, int> > Edges;    

protected:
  Nodes m_Nodes;
  Edges m_Edges;
	NodeEdges m_NodeEdges;
  NodeIndex m_NodeIndexes;
  bool	m_bGraphModified;

public:
  GraphIndexed()
  {    
    m_bGraphModified = false;
  }

public:
  bool Load(const std::string& filename)
  {    
    if (filename.rfind(".edge") == (filename.length() - 5))
	  {		  
      std::ifstream is;
		  is.open(filename.c_str());

		  //check
		  if(!is.good())
		  {
			  std::cout << " Graph failed to load: " << filename << std::endl;			  
			  return false;
		  }

		  int nnodes, nedges, id1, id2;

		  is >> nnodes >> nedges;
		  while((is >> id1 >> id2) && !is.eof())
		  {		
			  AddEdge(id1, id2);			
		  }

		  is.close();

		  // post check.		
 		  if (NumberOfEdges()!=nedges || NumberOfNodes()!=nnodes)
 		  {
        std::cerr<< "Error loading graphs, number of vertices " << NumberOfNodes() <<
				  " or edges "<< NumberOfEdges() <<" don't match! " <<std::endl; 			  
 			  return false;
 		  }		      
    }
    
    //std::cout<< "Graph loaded successfully: " << filename << std::endl;
    return true;
  }

public:
  int AddEdge(int nd1, int nd2)
  { 
    int idx1,idx2;
    int eg = (int)m_Edges.size();

    NodeIndex::iterator iter;    
    if((iter=m_NodeIndexes.find(nd1))==m_NodeIndexes.end())
    {
      idx1 = (int)m_Nodes.size();
      m_Nodes.push_back(nd1);
      m_NodeIndexes[nd1] = idx1;
      std::vector<int> tmp;
      tmp.push_back(eg);
      m_NodeEdges.push_back(tmp);
    }
    else
    {
      idx1 = (*iter).second;
      m_NodeEdges[idx1].push_back(eg);
    }

    if((iter=m_NodeIndexes.find(nd2))==m_NodeIndexes.end())
    {
      idx2 = (int)m_Nodes.size();
      m_Nodes.push_back(nd2);
      m_NodeIndexes[nd2] = idx2;
      std::vector<int> tmp;
      tmp.push_back(eg);
      m_NodeEdges.push_back(tmp);
    }
    else
    {
      idx2 = (*iter).second;
      m_NodeEdges[idx2].push_back(eg);
    }
    
    std::pair<int, int> tmp(idx1, idx2);
	  m_Edges.push_back(tmp);
    
    return eg;
  }

  int NumberOfNodes(void)const 
  { return (int)m_Nodes.size();}
	int NumberOfEdges(void)const 
  { return (int)m_Edges.size();}

  const Nodes & GetNodes()const
  { return m_Nodes; }
  const Nodes & GetNodes()
  { return m_Nodes; }

  const Edges & GetEdges()const
  { return m_Edges; }
  const Edges & GetEdges()
  { return m_Edges; }

  const NodeEdge & GetEdges(int n) 
  { return m_NodeEdges[n]; }
  const NodeEdge & GetEdges(int n)const
  { return m_NodeEdges[n]; }

  Nodes  GetNodes(int n)
  {
    const NodeEdge & nes = GetEdges(n);
    NodeEdge::const_iterator iter;
    Nodes nds;
    for(iter = nes.begin(); iter!=nes.end(); iter++ )
    {
      int nd1 = m_Edges[(*iter)].first;
      int nd2 = m_Edges[(*iter)].second;
      nds.push_back((nd1==n)? nd2 : nd1);
    }
    return nds;
  }
  Nodes  GetNodes(int n)const
  {
    const NodeEdge & nes = GetEdges(n);
    NodeEdge::const_iterator iter;
    Nodes nds;
    for(iter = nes.begin(); iter!=nes.end(); iter++ )
    {
      int nd1 = m_Edges[(*iter)].first;
      int nd2 = m_Edges[(*iter)].second;
      nds.push_back((nd1==n)? nd2 : nd1);
    }
    return nds;
  }

  int GetEdge(int n1, int n2)
  {
    const NodeEdge & nes = GetEdges(n1);
    NodeEdge::const_iterator iter;
    for(iter = nes.begin(); iter!=nes.end(); iter++ )
    {
      int nd1 = m_Edges[(*iter)].first;
      int nd2 = m_Edges[(*iter)].second;
      if(nd1 == n2 || nd2 == n2)
        return (*iter);
    }
    return -1;
  }
};

#endif //_GRAPH_INDEXED_H_
