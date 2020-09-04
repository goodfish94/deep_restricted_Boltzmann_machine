//
//  Ham.hpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/12.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#ifndef Ham_hpp
#define Ham_hpp

#include "libaray.h"

class Ham
{
public:
    Ham(int N_lattice, std::string Ham_type);
    std:: vector< double > element;
    std:: vector< int> index1;
    std:: vector< int> index2;
    
    double get_element(std::vector< std::vector< int > >  &n1, std::vector< std::vector< int > > &n2);
    
    
    
    std::string Ham_type;
    
    int N;// size of lattice;
    
    //int get_index( std::vector< std::vector< int > > &n );// get the index from a spin config.
    
    double Jz,h;
};

#endif /* Ham_hpp */
