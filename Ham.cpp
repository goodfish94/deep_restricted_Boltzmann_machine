//
//  Ham.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/12.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "Ham.hpp"


Ham::Ham(int N_lattice, std::string Ham_type_input)
{

    N=N_lattice;
    Ham_type=Ham_type_input;
    
    Jz=-1.0;
    
    h= 0.2;
    
    index1.resize(0);
    index2.resize(0);
    element.resize(0);

    
    
}

double Ham::get_element(std::vector< std::vector< int > >  &n1, std::vector< std::vector< int > > &n2)
{
    int i,j;
    int flip=0;
    
    double re=0.0;
    
    
    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            if(n1[i][j] != n2[i][j] )
            {
                flip = flip+1;
                if(flip > 1)
                {
                    return 0.0;
                }
              
            }
            
        }
    }
    
    
    if(flip>0)
    {
        return h/double(N*N);
    }
 
    //std::cout<<N<<" "<<Jz<<" "<<h<<std::endl;
    for(i=0;i<N-1;i++)
    {
        for(j=0;j<N-1;j++)
        {
            re = re + 4.0* Jz * (double(n1[i][j])-0.5) * (double(n1[i+1][j])-0.5)+4.0* Jz * (double(n1[i][j])-0.5) * (double(n1[i][j+1])-0.5);
        }
    }
    //std::cout<<re<<std::endl;
    for(i=0;i<N-1;i++)
    {
        re = re + 4.0* Jz * (double(n1[i][N-1])-0.5) * (double(n1[i+1][N-1])-0.5)+4.0* Jz * (double(n1[i][N-1])-0.5) * (double(n1[i][0])-0.5);
    }
    //std::cout<<re<<std::endl;
    
    for(j=0;j<N-1;j++)
    {
        re = re + 4.0* Jz * (double(n1[N-1][j])-0.5) * (double(n1[N-1][j+1])-0.5)+4.0* Jz * (double(n1[N-1][j])-0.5) * (double(n1[0][j])-0.5);
    }
    //std::cout<<re<<std::endl;
    
    re =re +4.0 * Jz * (double(n1[N-1][N-1]) -0.5 ) * (double(n1[0][N-1]) -0.5 ) +4.0 * Jz * (double(n1[N-1][N-1]) -0.5 ) * (double(n1[N-1][0]) -0.5 ) ;

    
    
    return re/(double(N*N));
    
}
