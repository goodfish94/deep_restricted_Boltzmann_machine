//
//  entanglement_entropy.hpp
//  neural_network_3
//
//  Created by huhaoyu on 2018/12/5.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#ifndef entanglement_entropy_hpp
#define entanglement_entropy_hpp

#include "libaray.h"
#include "DRM.hpp"

double get_entangelment_entropy( DRM & network, int N_visible, int N_block, std::vector<int> N_hidden);

double get_entangelment_entropy_fix_block( int block_a, DRM & network, int N_visible, int N_block, std::vector<int> N_hidden);
#endif /* entanglement_entropy_hpp */
