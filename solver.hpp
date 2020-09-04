//
//  solver.hpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/13.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#ifndef solver_hpp
#define solver_hpp

#include "libaray.h"
#include "DRM.hpp"
#include "Observable.hpp"
#include "entanglement_entropy.hpp"

class solver
{
    
public:
    
    //solver( std::string input_file,int N_block, int N_visible, std::vector<int> N_hidden, std::vector<std::string> parameter_file, std::vector<std::string> state_file , int p_minimum_in);
    
    solver( int N_block, int N_visible, std::vector<int> N_hidden,double p_minimum_in);//random initialize
    
    int solve();
    
    int write_energy(int sweep);
    
    int write_network( std::string filename);
    
    double cal_en_3lay();
    
    
private:
    
    Observable obs;
    DRM network;
    
    
    
    double lambda;
    double p_minimum;
    
    int freq_measure; // number of sampling during two sweep
    int n_sample;// number of sampling
    int n_sweep;// total sweep
    int n_warmup;// warm up
};

#endif /* solver_hpp */
