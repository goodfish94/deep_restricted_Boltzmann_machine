//
//  entanglement_entropy.cpp
//  neural_network_3
//
//  Created by huhaoyu on 2018/12/5.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "entanglement_entropy.hpp"


int generate_h_sate_ee( std::vector< std::vector<int> > &h_state, std::vector< int > N_hidden, int ind )
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

std::complex<double>  get_product( std::vector<int> &h1, std::vector<int> &h2,int if_b, int if_conj_1,int if_conj_2, DRM & network )
{
    std::complex<double> re =1.0, p1,p2;
    
    int indx,indy;
    
    
    for( int x=0 ;x<network.N_visible/2 ;x++)
    {
        
        if( if_b ==1 )
        {
            indx = network.N_visible-x-1;
        }
        else
        {
            indx = x;
        }
        indx = indx % network.N_block;
        for( int y=0; y<network.N_visible ;y++)
        {
            indy=y;
            indy = y % network.N_block;
            
            p1=0.0;
            p2=0.0;
            
            for( int i=0 ; i<network.N_hidden[0] ;i++)
            {
                p1 = p1 +  network.h_layer.weight[indx][indy][i] * double( h1[i] ) / double( network.N_visible * network.N_visible);
                p2 = p2 +  network.h_layer.weight[indx][indy][i] * double( h2[i] ) / double( network.N_visible * network.N_visible);
                
                
            }
            
            p1 = p1 + network.h_layer.bias_a[indx][indy] / double(  network.N_visible * network.N_visible );
            p2 = p2 + network.h_layer.bias_a[indx][indy] / double(  network.N_visible * network.N_visible );
            
            if(if_conj_1 == 1)
            {
                p1 = std::conj(p1);
            }
            if( if_conj_2 == 1)
            {
                p2 = std::conj(p2);
            }
            
            
            re = re * ( std::exp( -p1 -p2 ) + 1.0 );
        }
        
        
    }
    
    return re;
    
    
    
    
    
    
    
}

std::complex<double>  get_product_A( std::vector<int> &h1, std::vector<int> &h2,DRM & network )
{
    std::complex<double> re =1.0, p1,p2;
    
    int indx,indy;
    
    
    for( int x=0 ;x<network.N_visible ;x++)
    {
        
       
        indx = x % network.N_block;
        for( int y=0; y<network.N_visible ;y++)
        {
            indy = y % network.N_block;
            
            p1=0.0;
            p2=0.0;
            
            for( int i=0 ; i<network.N_hidden[0] ;i++)
            {
                p1 = p1 +  network.h_layer.weight[indx][indy][i] * double( h1[i] ) / double( network.N_visible * network.N_visible);
                p2 = p2 +  network.h_layer.weight[indx][indy][i] * double( h2[i] ) / double( network.N_visible * network.N_visible);
                
                
            }
            
            p1 = p1 + network.h_layer.bias_a[indx][indy] / double(  network.N_visible * network.N_visible );
            p2 = p2 + network.h_layer.bias_a[indx][indy] / double(  network.N_visible * network.N_visible );
            
            
            p2 = std::conj(p2);
            
            
            re = re * ( std::exp( -p1 -p2 ) + 1.0 );
        }
        
        
    }
    
    return re;
    
    
    
    
    
    
    
}

double get_entangelment_entropy( DRM & network, int N_visible, int N_block, std::vector<int> N_hidden)
{
    int N_total=0;
    
    int i;
    
    int i1,i2,i3,i4;
    
    std::complex<double> Asq,trsq;
    
    std::complex<double> p14,p23,p12,p34;
    
    std::vector< std::vector<int> > h1,h2,h3,h4;
    std::complex<double> p1,p2,p3,p4;
    
    for( i=0; i<int( N_hidden.size()) ;  i++)
    {
        N_total = N_total + N_hidden[i];
    }
    
    N_total = std::pow(2, N_total );
    
    h1.resize(int( N_hidden.size() ));
    h2.resize(int( N_hidden.size() ));
    h3.resize(int( N_hidden.size() ));
    h4.resize(int( N_hidden.size() ));
    
    for( i=0; i<int (N_hidden.size()) ; i++)
    {
        h1[i].resize(N_hidden[i]);
        h2[i].resize(N_hidden[i]);
        h3[i].resize(N_hidden[i]);
        h4[i].resize(N_hidden[i]);
        
    }
    
    for( i1=0; i1<N_total ;i1++ )
    {
        
        generate_h_sate_ee(h1, N_hidden, i1);
        p1 = network.return_p_hid(h1);
        
        for( i2=0; i2<N_total ;i2++)
        {
            generate_h_sate_ee(h2, N_hidden, i2);
            p2 = network.return_p_hid(h2);
            
            p12 = get_product( h1[0],h2[0], 1, 0 ,1,network );
            
            Asq = Asq + p1 * std::conj(p2) * get_product_A(h1[0], h2[0], network);
            
            for(i3=0;i3<N_total ;i3++)
            {
      
                generate_h_sate_ee(h3, N_hidden, i3);
                p3 = network.return_p_hid(h3);
                
                
                
                p23 = get_product( h2[0], h3[0], 0 , 1,0,network );
                
                
                
                for(i4=0;i4<N_total ;i4++)
                {
                    p14 = get_product( h1[0],h4[0],0,0,1, network);
                    p34 = get_product( h3[0],h4[0],1,0,1, network);
                    
      
                    
                    generate_h_sate_ee(h4, N_hidden, i4);
                    p4 = network.return_p_hid(h4);
                    
                    
                    
                    trsq = trsq + p1 * p3 * std::conj( p2 * p4 ) * p14 * p23 * p12 * p34;
                    
                }
                
                
                
            }
            
            
            
            
        }
        
        
        
        
        
    }
    
    trsq = trsq / Asq /Asq;
    //std::cout<<std::endl<<trsq<<" "<<Asq<<std::endl;
    //std::cout<<"{"<<N_visible*N_visible<<" , "<<-std::log( trsq.real() )<<"},";
    
    return trsq.real();
    
    
  
    
    
    
    
    
    
    
    
    
}




std::complex<double>  get_product_fix_block( int block_a,std::vector<int> &h1, std::vector<int> &h2,int if_b, int if_conj_1,int if_conj_2, DRM & network )
{
    std::complex<double> re =1.0, p1,p2;
    
    int indx,indy;
    
    
    for( int x=0 ;x<network.N_visible ;x++)
    {
        
        
        indx = x % network.N_block;
        for( int y=0; y<network.N_visible ;y++)
        {
            if( if_b ==1 )
            {
                if( y <= block_a && x<=block_a)
                {
                    continue;
                }
            }
            else
            {
                if( ! (y <= block_a && x<=block_a) )
                {
                    continue;
                }
            }
            
            indy = y % network.N_block;
            
            p1=0.0;
            p2=0.0;
            
            for( int i=0 ; i<network.N_hidden[0] ;i++)
            {
                p1 = p1 +  network.h_layer.weight[indx][indy][i] * double( h1[i] ) / double( network.N_visible * network.N_visible);
                p2 = p2 +  network.h_layer.weight[indx][indy][i] * double( h2[i] ) / double( network.N_visible * network.N_visible);
                
                
            }
            
            p1 = p1 + network.h_layer.bias_a[indx][indy] / double(  network.N_visible * network.N_visible );
            p2 = p2 + network.h_layer.bias_a[indx][indy] / double(  network.N_visible * network.N_visible );
            
            if(if_conj_1 == 1)
            {
                p1 = std::conj(p1);
            }
            if( if_conj_2 == 1)
            {
                p2 = std::conj(p2);
            }
            
            
            re = re * ( std::exp( -p1 -p2 ) + 1.0 );
        }
        
        
    }
    
    return re;
    
    
    
    
    
    
    
}



double get_entangelment_entropy_fix_block( int block_a, DRM & network, int N_visible, int N_block, std::vector<int> N_hidden)
{
    
    
    int N_total=0;
    
    int i;
    
    int i1,i2,i3,i4;
    
    std::complex<double> Asq,trsq;
    
    std::complex<double> p14,p23,p12,p34;
    
    std::vector< std::vector<int> > h1,h2,h3,h4;
    std::complex<double> p1,p2,p3,p4;
    
    for( i=0; i<int( N_hidden.size()) ;  i++)
    {
        N_total = N_total + N_hidden[i];
    }
    
    N_total = std::pow(2, N_total );
    
    h1.resize(int( N_hidden.size() ));
    h2.resize(int( N_hidden.size() ));
    h3.resize(int( N_hidden.size() ));
    h4.resize(int( N_hidden.size() ));
    
    for( i=0; i<int (N_hidden.size()) ; i++)
    {
        h1[i].resize(N_hidden[i]);
        h2[i].resize(N_hidden[i]);
        h3[i].resize(N_hidden[i]);
        h4[i].resize(N_hidden[i]);
        
    }
    
    for( i1=0; i1<N_total ;i1++ )
    {
        
        generate_h_sate_ee(h1, N_hidden, i1);
        p1 = network.return_p_hid(h1);
        
        for( i2=0; i2<N_total ;i2++)
        {
            generate_h_sate_ee(h2, N_hidden, i2);
            p2 = network.return_p_hid(h2);
            
            p12 = get_product_fix_block(block_a, h1[0],h2[0], 1, 0 ,1,network );
            
            Asq = Asq + p1 * std::conj(p2) * get_product_A(h1[0], h2[0], network);
            
            for(i3=0;i3<N_total ;i3++)
            {
                
                generate_h_sate_ee(h3, N_hidden, i3);
                p3 = network.return_p_hid(h3);
                
                
                
                p23 = get_product_fix_block(block_a, h2[0], h3[0], 0 , 1,0,network );
                
                
                
                for(i4=0;i4<N_total ;i4++)
                {
                
                    
                    
                    generate_h_sate_ee(h4, N_hidden, i4);
                    p4 = network.return_p_hid(h4);
                    
                    
                    p14 = get_product_fix_block(block_a, h1[0],h4[0],0,0,1, network);
                    p34 = get_product_fix_block(block_a, h3[0],h4[0],1,0,1, network);
                    
                    
                    trsq = trsq + p1 * p3 * std::conj( p2 * p4 ) * p14 * p23 * p12 * p34;
                    
                }
                
                
                
            }
            
            
            
            
        }
        
        
        
        
        
    }
    
    trsq = trsq / Asq /Asq;
    //std::cout<<std::endl<<trsq<<" "<<Asq<<std::endl;
    std::cout<<block_a*block_a<<"  "<<-std::log( trsq.real() )<<std::endl;
    
    return trsq.real();
    
    
    
    
    
    
    
    
    
    
}
