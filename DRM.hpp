//
//  DRM.hpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/11.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#ifndef DRM_hpp
#define DRM_hpp

#include "libaray.h"
#include "RBM_1D.hpp"
#include "RBM_2D.hpp"




class DRM
{//deep boltzman machine
public:
    
    
   // DRM(int N_block, int N_visible, std::vector<int> N_hidden, std::vector<std::string> parameter_file ,std::vector<std::string> state_file);
    
    DRM(int N_block, int N_visible, std::vector<int> N_hidden);// set parameters and state random
    
    //int load_state_from_file(std::vector<std::string> state_file);
   //int load_from_file(int N_block, int N_visible, std::vector<int> N_hidden, std::vector<std::string> parameter_file ,std::vector<std::string> state_file);
    
    int write_network( std::string filename);
    
    
    int initialize(); // initialzie p matrix for all the layers, visible layer, observable
    
    double sigmoid_n(  int i, int j); // flip (i,j) spin for state_ind, return sigmoid
    int update_n( int i, int j); // flip (i,j) spin for state_ind
    
    double sigmoid_h( int depth, int k);// flip k the unit of "depth"_th hidden layer return sigmoid
    int update_h(  int depth, int k);// flip k the unit of "depth"_th hidden layer return accept probability
    
    
    int update_RBM_2D( std::vector< std::vector< std::vector< std::complex<double> > > > &del_weight,  std:: vector< std::vector <std::complex<double> > > &del_bias_a,std::vector<std::complex<double> >  &del_bias_b, double lambda );
    // update the parameters for RBM_2D
    
    int update_RBM_1D( int d, std::vector< std::vector< std::complex<double> > >  &del_weight , std::vector<std::complex<double> >  &del_bias_b, double lambda );
    // update the parameters for dth RBM_1D
    
    
    int gibbs_sampling( );// sample and update all the units
    
    
    
    
    
    
    std::complex<double> return_p_state_1()
    {//return p for state_1;
        return p_1;
    }
  
    
  
    std::complex<double> return_p( std::vector< std::vector <int> > &state, std::vector< std::vector <int> > &h_state );
    // return p for certain states
    std::complex<double> return_p_hid( std::vector< std::vector <int> > &h_state );
    
    
    
   
    
    
    int N_block;
    int N_visible;
    std::vector<int> N_hidden;
    
    
    std::vector< std::vector <int> > state_1;// state for the visible layer
    std::vector< std::vector <int> > h_state_1;// state for all the hidden layers
    
    std::vector< std::vector< std::complex<double> > > x_state_1;// probabiliy ~sigmoid(x_state_1)
    std::vector< std::vector< std::complex<double> > > x_h_1;// probabiliy ~sigmoid(x_h_1)
    

    
    std::complex<double> p_1; // p(n_1,h_1,d_1)

    
    
    RBM_2D h_layer; //first hidden layer
    std::vector<RBM_1D> d_vector; // deep layer
    
    
    int update_h_fisrt_hidden_layer( int j ); // update the unit for 1st hidden layer
    
    int update_h_deep_hidden_layer( int i, int j ); // update the unit for 1st hidden layer
    
    
    
    
    // ----------generate random number ---------------------------
    
    int seed;
    double random_num();// generate random number
    
    
    int para_random();// randomly set the parameters of the network;
    int state_random();// randomly set the states of the network;
    int para_const(std::complex<double> constant);
    
    //---------test------------------
    friend int test_drm();
    friend int test_gibbs();
    
    std::complex<double> return_real_pn_3layer( int indi, int indj);// return probability for p=1 after trace out hidden
    
    
    
    
};

#endif /* DRM_hpp */
