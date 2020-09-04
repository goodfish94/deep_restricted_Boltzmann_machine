//
//  RBM_1D.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/11.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "RBM_1D.hpp"




RBM_1D::RBM_1D( )
{
    
    n_hidden=0;
    n_visible=0;
    
    clear_vector();
    
}

RBM_1D::~RBM_1D()
{
    n_hidden=0;
    n_visible=0;
    
    clear_vector();
}


int RBM_1D::allocate(int n_hidden_in, int n_visible_in)
{
    clear_vector();
    n_hidden = n_hidden_in;
    n_visible = n_visible_in;
    
    bias_b.resize(n_hidden,0.0);
    
    
    weight.resize(n_visible );
    for( int i=0 ;i< n_visible ; i++)
    {
     
      
        
        weight[i].resize( ( unsigned int )n_hidden,0.0);
        
    }
    
    return 0;
}

int RBM_1D::clear_vector()
{
    
    
    n_hidden=0;
    n_visible=0;
    
    std::vector< std::vector< std::complex<double> > >().swap(weight);
    
    
    std::vector< std::complex<double> >().swap(bias_b);
    
  
    
    return 0;
    
    
}

int RBM_1D::update_para( std::vector< std::vector< std::complex<double> > >  &del_weight, std::vector<std::complex<double> >  &del_bias_b, double lambda )
{
    
    int i,k;
    
    
    for(i=0;i<n_visible;i++)
    {
        
        
        for(k=0;k<n_hidden;k++)
        {
            weight[i][k] = weight[i][k] - lambda *std::conj( del_weight[i][k] );
        }
        
    }
    for(k=0;k<n_hidden;k++)
    {
        bias_b[k] = bias_b[k] - lambda * std::conj( del_bias_b[k] );
    }
    
    return 0;
    
}

