//
//  main.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/9.
//  Copyright © 2018 huhaoyu. All rights reserved.

#include "solver.hpp"


int cout_ee( int i , int N_lattice, int N_block,std::vector<int> N_hidden)
{

    std::complex<double> unit(0.0,1.0);
    DRM network(N_block, N_lattice,N_hidden);
    network.para_const(0.78539816339*unit);
    //network.para_random();
    get_entangelment_entropy_fix_block( i, network, N_lattice , N_block, N_hidden);
    //get_entangelment_entropy( network, 2*i , 2, N_hidden);
    return 1;
}

int main()
{
    // insert code here...
    int N_block = 0;
    int N_visible = 0;
    std::vector< int > N_hidden;
    int dep;

    
    std::fstream file;
    
    file.open( "network_size.txt" , std::ios::in);
    
    if( !file.is_open())
    {
        std::cout<<"no network size file "<<std::endl;
        exit(4);
    }
    
    file>>N_visible>>N_block;
    
    file>>dep;
    
    N_hidden.resize(dep);
    
    for( int i=0; i<dep ; i++)
    {
        file>>N_hidden[i] ;
    }
    
    file.close();
    
    
    
    double p_minimum_in = 0.0001;

    
   
    
    solver solve( N_block, N_visible, N_hidden, p_minimum_in);
    
    
    
    solve.solve();
    
    
    return 0;
}

