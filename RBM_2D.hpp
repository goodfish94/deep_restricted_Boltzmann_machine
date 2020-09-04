//
//  rbm.hpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/9.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

// RBM, visible layer 2D, hidden layer 1D, translational invariant

#ifndef rbm_hpp
#define rbm_hpp

#include "libaray.h"


class RBM_2D
{
    
public:
    
    RBM_2D(  int n_hidden_in, int n_block_in, std::string filename ); // read the file to initialize
    RBM_2D( ); 
    ~RBM_2D();
    
    int allocate(int n_hidden_in, int n_block_in);// allocate weight and bias
    //int load_from_file( int n_hidden_in, int n_block_in,  std::string filename );// load from file
    
    int clear_vector();// clear all the vector 
    

    
    int update_para(std::vector< std::vector< std::vector< std::complex<double> > > > &del_weight,  std:: vector< std::vector <std::complex<double> > > &del_bias_a,std::vector<std::complex<double> >  &del_bias_b, double lambda );// update parameter and p matrix
    
    
    //std:: vector< std::vector <std::vector < std::complex<double> > > > E_matrix;
    

    
    int n_hidden;
    //int n_visible; //n_visible^2 units
    int n_block;// translational invariant block n_tran * n_tran; visible unit
    
    std::vector< std::vector< std::vector< std::complex<double> > > >  weight;
    std:: vector< std::vector <std::complex<double> > > bias_a;
    std:: vector< std::complex<double>  > bias_b;
    
    //std:: vector< std::vector <std::vector < std::complex<double> > > > p_matrix;
    //store \exp(-(v*W*h+a*v+b*h))
    
    
};


#endif /* rbm_hpp */
