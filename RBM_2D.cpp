//
//  rbm.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/9.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "RBM_2D.hpp"




RBM_2D::RBM_2D()
{// random initialize
   // n_visible=0;
    n_hidden=0;
    n_block=0;
    clear_vector();
    
    
    
}

RBM_2D::~RBM_2D()
{
    n_hidden=0;
    n_block=0;
    //n_visible=0;
    
    clear_vector();
}


int RBM_2D::allocate(int n_hidden_in, int n_block_in)
{
    clear_vector();
   
    n_hidden = n_hidden_in;
    
    n_block = n_block_in;
    
   
    
    bias_a.resize(n_block  );
    
    for( int i=0; i<n_block; i++)
    {
        bias_a[i].resize(n_block,0.0);
       
    }
    
    bias_b.resize(n_hidden,0.0);
    
    
   // weight.resize(n_block, std::vector <std::vector<std::complex<double> > > (n_block, std::vector<std::complex<double> >(n_hidden,0.0) ) );
    
    weight.resize(n_block);
    
    for( int i=0; i<n_block; i++)
    {
        weight[i].resize(n_block);
        for( int j=0;j<n_block ; j++)
        {
            weight[i][j].resize(n_hidden, 0.0);
            
        }
    }
    
    return 1;
}




int RBM_2D::update_para(std::vector< std::vector< std::vector< std::complex<double> > > > &del_weight,  std:: vector< std::vector <std::complex<double> > > &del_bias_a,std::vector<std::complex<double> >  &del_bias_b, double lambda )
{
    
    int i,j,k;
    
    for(i=0;i<n_block;i++)
    {
        for(j=0;j<n_block;j++)
        {
            bias_a[i][j] = bias_a[i][j] - lambda * std::conj(del_bias_a[i][j]);
            for(k=0;k<n_hidden;k++)
            {
                weight[i][j][k] = weight[i][j][k] - lambda * std::conj(del_weight[i][j][k]);
            }
        }
    }
    for(k=0;k<n_hidden;k++)
    {
        bias_b[k] = bias_b[k] - lambda * std::conj(del_bias_b[k]);
    }
    
    return 0;
    
}


int RBM_2D::clear_vector()
{

    n_hidden=0;
    n_block=0;

    
    std::vector< std::vector< std:: vector< std::complex<double> > > >().swap(weight);
    
    std::vector<  std:: vector< std::complex<double> > > ().swap(bias_a);
    
    std::vector< std::complex<double> >().swap(bias_b);
    
    //std::vector< std::vector< std::vector< std::complex<double> > > >().swap(p_matrix);
    //std::vector< std::vector< std::vector< std::complex<double> > > >().swap(E_matrix);
    
    return 0;
    
    
}


