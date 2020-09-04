//
//  Observable.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/12.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "Observable.hpp"

Observable::Observable( std::string Ham_type, int N_block_in, int N_visible_in, std::vector<int> N_hidden_in, double minimum_p_in):Hamiltonian(N_visible_in,Ham_type)
{
    
    minimum_p = minimum_p_in;
    
    N_block = N_block_in;
    N_visible = N_visible_in;
    N_hidden = N_hidden_in;
    
    O_D_weight_2D.resize(N_block );
    for( int i=0; i<N_block ; i ++)
    {
        O_D_weight_2D[i].resize(N_block);
        for( int j=0; j<N_block ; j++)
        {
            O_D_weight_2D[i][j].resize(N_hidden[0],0.0);
            
        }
        
    }
  
    I_D_weight_2D.resize(N_block );
    
    for( int i=0; i<N_block ; i ++)
    {
        I_D_weight_2D[i].resize(N_block);
        for( int j=0; j<N_block ; j++)
        {
            I_D_weight_2D[i][j].resize(N_hidden[0],0.0);
            
        }
        
    }
    
    partial_weight_2D.resize(N_block);
    
    for( int i=0; i<N_block ; i ++)
    {
        partial_weight_2D[i].resize(N_block);
        for( int j=0; j<N_block ; j++)
        {
            partial_weight_2D[i][j].resize(N_hidden[0],0.0);
            
        }
        
    }
    
    O_D_bias_a_2D.resize(N_block);
    for( int i=0 ; i< N_block ; i++)
    {
        O_D_bias_a_2D[i].resize(N_block, 0.0);
    }
   
    I_D_bias_a_2D.resize(N_block);
    for( int i=0 ; i< N_block ; i++)
    {
        I_D_bias_a_2D[i].resize(N_block, 0.0);
    }
    
    partial_bias_a_2D.resize(N_block );
    for( int i=0 ; i< N_block ; i++)
    {
        partial_bias_a_2D[i].resize(N_block, 0.0);
    }
    
    O_D_bias_b_2D.resize(N_hidden[0] ,0.0);
    I_D_bias_b_2D.resize(N_hidden[0] ,0.0);
    partial_bias_b_2D.resize(N_hidden[0],0.0 );
    
    int num_of_layer = int(N_hidden_in.size())-1;
    
    
    O_D_weight_1D.resize(num_of_layer);
    I_D_weight_1D.resize(num_of_layer);
    partial_weight_1D.resize(num_of_layer);
    
    O_D_bias_b_1D.resize(num_of_layer);
    I_D_bias_b_1D.resize(num_of_layer);
    partial_bias_b_1D.resize(num_of_layer);
    
    for( int dep=0; dep <num_of_layer ; dep++)
    {
        
        O_D_weight_1D[dep].resize( N_hidden[dep]  );
        for( int i=0; i< N_hidden[dep] ; i++)
        {
            O_D_weight_1D[dep][i].resize(N_hidden[dep+1] , 0.0 );
        }
        
        I_D_weight_1D[dep].resize( N_hidden[dep]);
        for( int i=0; i< N_hidden[dep] ; i++)
        {
            I_D_weight_1D[dep][i].resize(N_hidden[dep+1] , 0.0 );
        }
        
        partial_weight_1D[dep].resize( N_hidden[dep]  );
        for( int i=0; i< N_hidden[dep] ; i++)
        {
            partial_weight_1D[dep][i].resize(N_hidden[dep+1] , 0.0 );
        }
        
        O_D_bias_b_1D[dep].resize(N_hidden[dep+1] , 0.0);
        I_D_bias_b_1D[dep].resize(N_hidden[dep+1] , 0.0);
        partial_bias_b_1D[dep].resize(N_hidden[dep+1] , 0.0);
        
    }
    
    
    state_2.resize( N_visible , std::vector< int > (N_visible, 0 ) );
    for( int i=0 ; i< N_visible ; i++)
    {
        state_2[i].resize(N_visible);
    }
    
    h_state_2.resize( int( N_hidden.size() ) );
    
    for( int i=0; i < int ( N_hidden.size() ) ; i++ )
    {
        h_state_2[i].resize(N_hidden[i],0);
        
        
    }
    
    
    set_zero();
    
 
    
    
    
}


int Observable::set_zero()
{// set all the obs zero
    
    int i,j,k,m,dep;
    
    n_measure=0;
    Htilde=0.0;
    Itilde=0.0;
    magnetization=0.0;
    
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block;j++)
        {
            for(k=0;k<N_hidden[0];k++)
            {
                O_D_weight_2D[i][j][k]=0.0;
                //dev_O_D_weight_2D[i][j][k]=0.0;
                I_D_weight_2D[i][j][k]=0.0;
                partial_weight_2D[i][j][k]=0.0;
                
                
            }
            O_D_bias_a_2D[i][j]=0.0;
            I_D_bias_a_2D[i][j]=0.0;
            partial_bias_a_2D[i][j]=0.0;
        }
    }
    for(k=0;k<N_hidden[0];k++)
    {
        O_D_bias_b_2D[k]=0.0;
        I_D_bias_b_2D[k]=0.0;
        partial_bias_b_2D[k]=0.0;
        
    }
    for( dep = 0; dep<int(N_hidden.size())-1 ;dep++)
    {
        for( m=0; m<N_hidden[dep+1] ; m++)
        {
            for( k=0; k <N_hidden[dep] ; k++)
            {
                O_D_weight_1D[dep][k][m] =  0.0;
                I_D_weight_1D[dep][k][m] =  0.0;
                partial_weight_1D[dep][k][m] = 0.0;
                
            }
            O_D_bias_b_1D[dep][m] = 0.0;
            I_D_bias_b_1D[dep][m] = 0.0;
            partial_bias_b_1D[dep][m] = 0.0;
        }
    }
    
    return 0;
        
}



int generate_h_sate( std::vector< std::vector<int> > &h_state, std::vector< int > N_hidden, int ind )
{
    
    int i,j;
    
    
    
    for( i=0 ;i < int(N_hidden.size()) ;i++ )
    {
        for( j=0; j<N_hidden [i] ; j++)
        {
            
            h_state[i][j] = ind % 2 ;
            ind = (ind-ind%2)/2;

            
            
        }
    }
    
    return 1;
    
    
}

int Observable::get_observable(std::vector< std::vector <int> > &state_1, std::vector< std::vector<int> > &h_state_1, std::complex<double> p1 , DRM & network )
{
    
    
    
    

    int i,j,k,m;
    int swp;
    int dep;
    
    int flipx, flipy;
    
    std::complex<double> h12,i12;
    std::complex<double> h;
    
    
    
    std::complex<double> p2, flip_p2;
    
    double del_n2;
    
    double mag;
    
    n_measure = n_measure + 1;
    
   
    
    if( std::abs(p1) <=minimum_p )
    {
        return 0.0;
    }
    
    
    
    
    dep = int (N_hidden.size());
    
    
    
    int total_hidden=0;
    
    for( i=0; i<int(N_hidden.size()) ;i++)
    {
        total_hidden = total_hidden + N_hidden[i];
    }
    
    state_2 = state_1;
    
    h=Hamiltonian.get_element(state_1, state_2);
    
    mag=0.0;
    for( i=0; i<N_visible ; i++)
    {
        for( j=0;j<N_visible ;j++)
        {
            
            mag = mag + double( state_1[i][j] );
        }
    }
    
    mag = mag/ double( N_visible * N_visible )-0.5;
    

    
    for( swp=0; swp<std::pow(2, total_hidden ); swp++)
    {
        
        
        //-------------------------------daignal contribution---------------------------------
        
        generate_h_sate(h_state_2, N_hidden, swp);
        
     
        
        p2 = network.return_p( state_2, h_state_2 );
        
       
        h12 = h * p2/p1;
        
        
        i12 = p2 / p1;
        
        Htilde = Htilde + h12;
        Itilde = Itilde + i12;
        magnetization = magnetization + mag * p2/p1;
        
        
        
        
        for(i=0;i<N_visible;i++)
        {
            for(j=0;j<N_visible ;j++)
            {
                
                for(k=0;k<N_hidden[0];k++)
                {
                    O_D_weight_2D[i%N_block][j%N_block][k] = O_D_weight_2D[i%N_block][j%N_block][k] - double(state_2[i][j] * h_state_2[0][k]) * h12/ double( N_visible * N_visible);
                    I_D_weight_2D[i%N_block][j%N_block][k] = I_D_weight_2D[i%N_block][j%N_block][k] - double(state_2[i][j] * h_state_2[0][k]) * i12/ double( N_visible * N_visible);
                    
         
                }
                O_D_bias_a_2D[i%N_block][j%N_block] = O_D_bias_a_2D[i%N_block][j%N_block] - double( state_2[i][j] )*h12/ double( N_visible * N_visible);
                I_D_bias_a_2D[i%N_block][j%N_block] = I_D_bias_a_2D[i%N_block][j%N_block] - double( state_2[i][j] )*i12/ double( N_visible * N_visible);
                
            }
            
        }
        
        for(k=0;k<N_hidden[0];k++)
        {
            O_D_bias_b_2D[k] = O_D_bias_b_2D[k] - double( h_state_2[0][k] ) * h12;
            I_D_bias_b_2D[k] = I_D_bias_b_2D[k] - double( h_state_2[0][k] ) * i12;
        }
        
        
        // for deep layer
        for( dep = 0 ; dep<int( N_hidden.size() )-1 ;dep ++ )
        {
            
            for( m=0; m<N_hidden[dep+1] ; m++)
            {
                for( k=0; k <N_hidden[dep] ; k++)
                {
                    O_D_weight_1D[dep][k][m] = O_D_weight_1D[dep][k][m] - double( h_state_2[dep][k] * h_state_2[dep+1][m]) * h12;
                    I_D_weight_1D[dep][k][m] = I_D_weight_1D[dep][k][m] - double( h_state_2[dep][k] * h_state_2[dep+1][m]) * i12;
                    
                }
                O_D_bias_b_1D[dep][m] = O_D_bias_b_1D[dep][m] - double( h_state_2[dep+1][m] )*h12;
                I_D_bias_b_1D[dep][m] = I_D_bias_b_1D[dep][m] - double( h_state_2[dep+1][m] )*i12;
            }
            
        }
        
        
         //------------------------------- off daignal contribution---------------------------------
        
        
        
        for ( flipx = 0 ; flipx < N_visible ; flipx ++ )
        {
            
            for ( flipy = 0 ; flipy < N_visible ; flipy ++ )
            {// flip one spin in state 2
                
                
                del_n2 = 1.0 - 2.0 * state_2[flipx][flipy];
                
                state_2[flipx][flipy] = 1- state_2[flipx][flipy];
                
                
                
                flip_p2 = 0.0;
                
                
                for( k=0; k<N_hidden[0] ; k++ )
                {
                    
                    
                    flip_p2 = flip_p2 + network.h_layer.weight[ (flipx % N_block) ][ flipy % N_block ][k]  * double ( h_state_2[0][k] ) * del_n2/ double( N_visible * N_visible);
                    
                    
                }
                
                flip_p2 = flip_p2 +  network.h_layer.bias_a[ flipx % N_block ] [ flipy % N_block ] * del_n2/ double( N_visible * N_visible);
                
                flip_p2 = p2 * std::exp( -flip_p2 );
                
                
                

                
                h12 = Hamiltonian.h/double(N_visible * N_visible) * flip_p2 / p1;
                
                Htilde = Htilde + h12 ;
                
                
                
                
                for(i=0;i<N_visible;i++)
                {
                    for(j=0;j<N_visible ;j++)
                    {
                        
                        for(k=0;k<N_hidden[0];k++)
                        {
                            O_D_weight_2D[i%N_block][j%N_block][k] = O_D_weight_2D[i%N_block][j%N_block][k] - double(state_2[i][j] * h_state_2[0][k]) * h12/ double( N_visible * N_visible);
    
                            
                            
                        }
                        O_D_bias_a_2D[i%N_block][j%N_block] = O_D_bias_a_2D[i%N_block][j%N_block] - double( state_2[i][j] )*h12/ double( N_visible * N_visible);
            
                        
                    }
                    
                }
                
                for(k=0;k<N_hidden[0];k++)
                {
                    O_D_bias_b_2D[k] = O_D_bias_b_2D[k] - double( h_state_2[0][k] ) * h12;
                }
                
                
                // for deep layer
                for( dep = 0 ; dep<int( N_hidden.size() )-1 ;dep ++ )
                {
                    
                    for( m=0; m<N_hidden[dep+1] ; m++)
                    {
                        for( k=0; k <N_hidden[dep] ; k++)
                        {
                            O_D_weight_1D[dep][k][m] =  O_D_weight_1D[dep][k][m] - double( h_state_2[dep][k] * h_state_2[dep+1][m]) * h12;
                            
                            
                        }
                        O_D_bias_b_1D[dep][m] = O_D_bias_b_1D[dep][m] - double( h_state_2[dep+1][m] )*h12;
                       
                    }
                    
                }
                
                
        
                
                
                state_2[flipx][flipy] = 1- state_2[flipx][flipy];
                
                
            }
            
        }
        
    }
    
    
    
    

    
    return 1;
    
    
}


 int Observable::get_average()
{// cal the expectation value
    //return 0 if Itilde < 0.0
    
    int i,j,k;
    
    int dep, m;
    
    if( std::abs(Itilde) < minimum_p )
    {
        return 0;
    }
    
    
    Htilde = Htilde/double(n_measure) ;
    Itilde = Itilde/double(n_measure) ;
    
    magnetization = magnetization / double( n_measure );
    
    
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block ;j++)
        {
            
            for(k=0;k<N_hidden[0];k++)
            {
                O_D_weight_2D[i][j][k] =O_D_weight_2D[i][j][k]/double(n_measure);
                //dev_O_D_weight_2D[i][j][k] =dev_O_D_weight_2D[i][j][k]/double(n_measure);
                I_D_weight_2D[i][j][k] =I_D_weight_2D[i][j][k]/double(n_measure);
                
                
            }
            
            O_D_bias_a_2D[i][j] = O_D_bias_a_2D[i][j]/double(n_measure);
            I_D_bias_a_2D[i][j] = I_D_bias_a_2D[i][j]/double(n_measure);
        }
        
    }
    
    for(k=0;k<N_hidden[0];k++)
    {
        O_D_bias_b_2D[k] = O_D_bias_b_2D[k] / double(n_measure);
        I_D_bias_b_2D[k] = I_D_bias_b_2D[k] / double(n_measure);
    }
    
    for( dep = 0 ; dep< int(N_hidden.size())-1 ;dep ++ )
    {
        
        for( m=0; m<N_hidden[dep+1] ; m++)
        {
            for( k=0; k <N_hidden[dep] ; k++)
            {
                O_D_weight_1D[dep][k][m] =  O_D_weight_1D[dep][k][m] / double(n_measure);
                I_D_weight_1D[dep][k][m] =  I_D_weight_1D[dep][k][m] / double(n_measure);
                
            }
            O_D_bias_b_1D[dep][m] = O_D_bias_b_1D[dep][m] / double(n_measure);
            I_D_bias_b_1D[dep][m] = I_D_bias_b_1D[dep][m] / double(n_measure);
        }
        
    }
    
    return 1;
    
    
}


int Observable::cal_partial()
{
  
    
    int i,j,k,m,dep;
    
    Htilde = Htilde.real();
    Itilde = Itilde.real();
    
    Energy = Htilde.real()/(Itilde.real());
    magnetization = magnetization.real() / Itilde.real();
    
    
    
   
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block;j++)
        {
            for(k=0;k<N_hidden[0];k++)
            {
                
                partial_weight_2D[i][j][k]= (O_D_weight_2D[i][j][k]*Itilde - I_D_weight_2D[i][j][k]*Htilde)/(Itilde * Itilde );
                
                
            }
            
            partial_bias_a_2D[i][j] = (O_D_bias_a_2D[i][j]*Itilde - I_D_bias_a_2D[i][j]*Htilde)/(Itilde * Itilde );
           
        }
    }
    
    for(k=0;k<N_hidden[0];k++)
    {
        partial_bias_b_2D[k] = (O_D_bias_b_2D[k]*Itilde - I_D_bias_b_2D[k]*Htilde)/(Itilde * Itilde );
        
    }
    
    for( dep = 0 ; dep< int(N_hidden.size())-1 ;dep ++ )
    {
        
        for( m=0; m<N_hidden[dep+1] ; m++)
        {
            for( k=0; k <N_hidden[dep] ; k++)
            {
                partial_weight_1D[dep][k][m] = (O_D_weight_1D[dep][k][m] * Itilde - I_D_weight_1D[dep][k][m] * Htilde)/( Itilde * Itilde );
                
                
            
                
            }
            
            partial_bias_b_1D[dep][m] = (O_D_bias_b_1D[dep][m] * Itilde - I_D_bias_b_1D[dep][m] * Htilde)/(Itilde * Itilde);
           
        }
        
    }
    
    
    return 1;
    
}
