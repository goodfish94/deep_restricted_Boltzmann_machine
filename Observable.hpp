//
//  Observable.hpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/12.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#ifndef Observable_hpp
#define Observable_hpp

#include "libaray.h"
#include "Ham.hpp"
#include "DRM.hpp"


class Observable
{
    
    int N_block;
    int N_visible;
    std::vector<int> N_hidden;
    
    Ham Hamiltonian;
    

    int n_measure;
    
    std::vector< std::vector< std::vector< std::complex<double> > > > partial_weight_2D;
    std::vector< std::vector< std::complex<double> > > partial_bias_a_2D;
    std::vector< std::complex<double> > partial_bias_b_2D;
    
    std::vector< std::vector< std::vector< std::complex<double> > > > partial_weight_1D;
    std::vector< std::vector< std::complex<double> > > partial_bias_b_1D;
    
    
    

    
    
    std::vector< std::vector< std::vector< std::complex<double> > > >  O_D_weight_2D;
    std::vector< std::vector< std::vector< std::complex<double> > > >  dev_O_D_weight_2D;
    std:: vector< std::vector <std::complex<double> > > O_D_bias_a_2D;
    std:: vector< std::complex<double>  > O_D_bias_b_2D;
    
    std::vector< std::vector< std::vector< std::complex<double> > > >  O_D_weight_1D;
    std:: vector< std::vector < std::complex<double> > > O_D_bias_b_1D;
    // tilde{O} * D
    
    std::vector< std::vector< std::vector< std::complex<double> > > >  I_D_weight_2D;
    std:: vector< std::vector <std::complex<double> > > I_D_bias_a_2D;
    std:: vector< std::complex<double>  > I_D_bias_b_2D;
    
    std::vector< std::vector< std::vector< std::complex<double> > > >  I_D_weight_1D;
    std:: vector< std::vector < std::complex<double> > > I_D_bias_b_1D;
    // tilde{I} * D
    
    std::complex<double> Itilde;
    std::complex<double> Htilde;
    
    double Energy;
    
    std::complex<double> magnetization;// magnetization
    
    double minimum_p;
    
    
public:
    
    Observable( std::string Ham_type, int N_block_in, int N_visible_in, std::vector<int> N_hidden_in, double minimum_p_in);
    
    int get_observable(std::vector< std::vector <int> > &state_1 ,std::vector< std::vector <int> > &h_state_1, std::complex<double> p1,DRM & network );// calculate energy and partial for the config
  
    int get_average();// cal the expectation value
    
    int cal_partial();// calculate the partial derivative

    int set_zero(); // set all the obs zero
    

    
    std::vector< std::vector <int> > state_2;
    std::vector< std::vector <int> > h_state_2;
    
    
    
    
    friend class solver;
  
    
   
    
};

#endif /* Observable_hpp */
