//
//  RBM_1D.hpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/11.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#ifndef RBM_1D_hpp
#define RBM_1D_hpp

#include "libaray.h"

// RBM hidden layer 1D, visible layer 1D, non-translational invraiant

class RBM_1D
{
    
public:
    
    RBM_1D( int n_hidden, int n_visible, std::string filename ); // read the file to initialize
    RBM_1D(  ); 
    ~RBM_1D();
    
    int allocate(int n_hidden, int n_visible);// allocate weight and bias b
    //int load_from_file( int n_hidden, int n_visible, std::string filename );// load from file
    
    int clear_vector();// clear all the vector
    
  
    
    int update_para( std::vector< std::vector< std::complex<double> > >  &del_weight,std::vector<std::complex<double> >  &del_bias_b, double lambda );
    // update parameter and p matrix
    
    
    
    
    int n_hidden;
    int n_visible; //n_visible units

    
    std::vector< std::vector< std::complex<double> >  >  weight;
    std:: vector< std::complex<double> > bias_b;
    
   // std::vector< std::vector < std::complex<double> > > p_matrix;
    //store \exp(-(v*W*h+a*v+b*h))
    
    
    //store (v*W*h+a*v+b*h)
    

    
    
};


#endif /* RBM_1D_hpp */
