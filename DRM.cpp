//
//  DRM.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/11.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "DRM.hpp"


DRM::DRM(int N_block_in, int N_visible_in, std::vector<int> N_hidden_in)
{
    unsigned int depth;
    int  i;

    N_block = N_block_in;
    
    N_visible = N_visible_in;
    
    N_hidden.resize(N_hidden_in.size());
    
    for( i=0 ; i< int( N_hidden_in.size() ); i++)
    {
        N_hidden[i] =  N_hidden_in[i];
    }
   
    
    
    
    
    
    
    depth = ( N_hidden_in.size() );
    
    if(depth<2)
    {
        std::cout<<"depth wrong for DRM"<<std::endl;
        exit(1);
    }
    
    h_layer.allocate( N_hidden_in[0], N_block_in);
    
    //std::cout<<"dep"<<depth - 1 ;
    
    d_vector.resize( depth-1 );
    
    for(i=1;i<depth; i++)
    {
        d_vector[i-1].allocate(N_hidden_in[i], N_hidden_in[i-1]);
       
    }
    
    N_block = h_layer.n_block;
    
    
    
    
    
    state_1.resize(N_visible  );
    for( i=0 ; i<N_visible ; i++ )
    {
        state_1[i].resize(N_visible,0 );
    }
   
   
    
    x_state_1.resize(N_block );
    
    for( i=0; i<N_block ; i++)
    {
        x_state_1[i].resize( N_block, 0.0);
    }
    
    h_state_1.resize(depth);
    
    
    for(i=0;i<depth;i++)
    {
     
        h_state_1[i].resize(N_hidden[i],0);
       
    }
    
   
    
    x_h_1.resize(depth);
   
    
    for(i=0;i<depth;i++)
    {
        x_h_1[i].resize(N_hidden[i],0);
       
        
    }
    
    
    seed=int( time(NULL) );
    seed = 1;
    srand(seed);
    
   
    
    para_random();
    
    
    state_random();
    
    
}





int DRM::initialize()
{// initialze x and p
    
    
    int i,j,k;
    
    std::complex<double> E1;
    
    
    
    
    // set x_state
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block;j++)
        {
            E1 = h_layer.bias_a[ i%N_block ][ j%N_block ] / double( N_visible * N_visible);
           
            
            for( k=0;k<N_hidden[0];k++)
            {
                
                
                E1 = E1 + h_layer.weight[ i%N_block ][ j%N_block ][k]*double(h_state_1[0][k])/ double( N_visible * N_visible);
                
            
                
            }
            
            x_state_1[i][j] = E1;
           
        }
    }
    
    // set x_h[0]
    for(i=0;i<N_hidden[0];i++)
    {
        E1 = h_layer.bias_b[i];
       


            
        for( k=0;k<N_hidden[1];k++)
        {
                
                
            E1 = E1 + d_vector[0].weight[ i ][k]*double(h_state_1[1][k]);
            
           
                
        }
        
        for( j=0;j<N_visible;j++)
        {
            for(k=0;k<N_visible;k++)
            {
                E1 = E1 + h_layer.weight[ j%N_block ][ k%N_block ][i] * double( state_1[j][k] )/double(N_visible * N_visible);
               
            }
        }
        
        x_h_1[0][i] = E1;
      
    }
    
    int index;
    int layer;
    
    for(  layer = 1 ; layer< int(N_hidden.size() -1 ) ;layer++ )
    {
        index = layer-1;
        for(i=0;i<N_hidden[layer];i++)
        {
            
            E1 = d_vector[index].bias_b[i] ;
           
            
            for(k=0;k<N_hidden[layer-1] ;k++ )
            {// previous layer
                E1 = E1 + d_vector[index].weight[ k ][i]*double(h_state_1[layer-1][k]);
                
               
            }
            
            for(k=0;k<N_hidden[layer+1] ;k++ )
            {// the layer after it
                E1 = E1 + d_vector[index+1].weight[ i ][k]*double(h_state_1[layer+1][k]);
                
               
            }
            
            x_h_1[layer][i] = E1;
            
            
        }
    }
    
    
    index = int(N_hidden.size())-2; // final layer
    layer = int(N_hidden.size())-1;
    
    for(i=0;i<N_hidden[layer];i++)
    {
        E1 = d_vector[index].bias_b[i];
     
        
        for(k=0;k<N_hidden[layer-1] ; k++)
        {
            E1 = E1 + d_vector[index].weight[k][i]*double(h_state_1[layer-1][k]);
          
        }
        
        x_h_1[layer][i] = E1;
        
    }
    

    
    //-------------SET P----------------
    
    E1=0.0;

    for(i=0;i<N_visible; i++)
    {
        for(j=0;j<N_visible; j++)
        {
            E1 = E1+ h_layer.bias_a[i%N_block][j%N_block] * double( state_1[i][j] )/ double( N_visible * N_visible) ;
           
            
            for(k=0;k<N_hidden[0];k++)
            {
                E1 = E1 + h_layer.weight[i%N_block][j%N_block][k] * double(state_1[i][j] * h_state_1[0][k] )/ double( N_visible * N_visible) ;
                
            }
            
        }
    }
    
    for(i=0;i<N_hidden[0];i++)
    {
        E1=E1 + h_layer.bias_b[i] * double( h_state_1[0][i] );
       
    }
    
    
    for(layer = 1; layer < int(N_hidden.size()) ; layer++)
    {
        
        index = layer-1;
        
        for(i=0; i<N_hidden[layer-1] ; i++)
        {
            for(j=0; j<N_hidden[layer] ; j++)
            {
                E1 = E1 + d_vector[index].weight[i][j] * double( h_state_1[layer-1][i] * h_state_1[layer][j] );
               
            }
        }
        
        for(i=0 ; i<N_hidden[layer] ; i++)
        {
            E1 = E1+ d_vector[index].bias_b[i] * double( h_state_1[layer][i] );
           
        }
        
        
        
    }
    
    p_1 = std::exp(-E1) ;
   
    
    
    
    
    return 1;

}




double DRM::sigmoid_n(  int i, int j)
{
    
    double real;
    
    real = 2.0 * x_state_1[i][j].real();
    
    
   
    return std::exp(-real )/(1.0 + std::exp(-real) );
    
}


double DRM::sigmoid_h( int depth, int k)
{// flip k the unit of "depth"_th hidden layer return accept probability
    
    double real;
    
    
    real = 2.0 * x_h_1[depth][k].real();
 
    
    return std::exp(-real )/(1.0 + std::exp(-real) );

}



int DRM::update_n( int i, int j)
{// flip (i,j) spin for state_ind

    
    int k=0;
    
    std::complex<double> del_E=0.0;
    std::complex<double> w_times_x;
    
    double del_n;
    
   
        state_1[i][j] = 1 - state_1[i][j];// update state;
        
        del_n = 2.0* ( double(state_1[i][j]) - 0.5);
        
        
        
        for(k=0; k<N_hidden[0] ; k++)
        {//update the x_h for the 1st hidden layer
            w_times_x =  h_layer.weight[i % N_block ][j % N_block][k] * del_n / double( N_visible * N_visible);
            x_h_1[0][k] = x_h_1[0][k] + w_times_x;
            del_E = del_E + w_times_x * double(h_state_1[0][k]);
        }
        
        del_E = del_E + del_n* h_layer.bias_a[i % N_block][j % N_block]/double( N_visible * N_visible);
        p_1 = p_1 * std::exp(-del_E);
        
        
    
    return 0;


}


int DRM::update_h( int i, int j)
{// flip (i,j) spin for state_ind
    
    
    if( i==0 )
    {
        update_h_fisrt_hidden_layer(j );
        
    }
    else
    {
         update_h_deep_hidden_layer( i, j );
    }
    
    return 0;
    
    
}

int DRM::update_h_fisrt_hidden_layer( int k )
{
    
    
    int i=0,j=0;
    
    std::complex<double> del_E=0.0;
    std::complex<double> w_times_h;
    double del_h;
    
   
        h_state_1[0][k] = 1 - h_state_1[0][k];// update state;
        
        del_h = 2.0*(double( h_state_1[0][k]) - 0.5);
        
  
        // update x_state
        for(i=0;i<N_block; i++)
        {
            for(j=0;j<N_block;j++)
            {
                w_times_h =  h_layer.weight[ i  ][ j ][k] * del_h/ double( N_visible * N_visible);
                
                x_state_1[i][j] = x_state_1[i][j] + w_times_h;
                
               
                
            }
        }
        
        for(i=0;i<N_visible;i++)
        {
            for(j=0;j<N_visible;j++)
            {
                w_times_h =  h_layer.weight[i % N_block ][j % N_block][k] * del_h/ double( N_visible * N_visible);
                del_E = del_E + w_times_h * double( state_1[i][j] );
            }
        }
        
        
        del_E = del_E + del_h * h_layer.bias_b[k];
    
        
        //update_h_state of second layer
        for(i=0; i<N_hidden[1] ; i++)
        {
         
            w_times_h =  d_vector[0].weight[k][i] * del_h;
            x_h_1[1][i] = x_h_1[1][i] + w_times_h;
            
            del_E =del_E + w_times_h * double( h_state_1[1][i]);
            
        }
        
        p_1 = p_1 * std::exp(-del_E);
        
        
        
   
    return 0;
    
}


int DRM::update_h_deep_hidden_layer( int dep, int k )
{
    
    
    int i=0;
    
    std::complex<double> del_E=0.0;
    std::complex<double> w_times_h;
    double del_h;
    
    
        h_state_1[dep][k] = 1 - h_state_1[dep][k];// update state;
        
        del_h = 2.0*(double( h_state_1[dep][k]) - 0.5);
    
    
    
        
        // update x_h for the previous layer
        for( i=0; i< N_hidden[dep-1] ;i++)
        {
            w_times_h = d_vector[dep-1].weight[i][k] * del_h;
            
            x_h_1[dep-1][i] = x_h_1[dep-1][i] + w_times_h;
            
            del_E = del_E + w_times_h * double( h_state_1[dep-1][i] );
            
            
        }
        
        del_E = del_E + d_vector[dep-1].bias_b[k]*del_h;
       
        
        
    
        
        if( dep == int( N_hidden.size() )-1 )
        {// if the final layer
            p_1 = p_1 * std::exp(-del_E);
            return 0;
            
        }
        
        //update x_h for the layer after it
        for(i=0; i<N_hidden[dep+1] ; i++)
        {
            
            w_times_h =  d_vector[dep].weight[k][i] * del_h;
            x_h_1[dep+1][i] = x_h_1[dep+1][i] + w_times_h;
            
            del_E =del_E + w_times_h * double( h_state_1[dep+1][i]);
            
        }
        
        p_1 = p_1 * std::exp(-del_E);
        

        
    
    return 0;
    
}



int DRM::update_RBM_2D( std::vector< std::vector< std::vector< std::complex<double> > > > &del_weight,  std:: vector< std::vector <std::complex<double> > > &del_bias_a,std::vector<std::complex<double> >  &del_bias_b, double lambda )
{
// update the parameters for RBM_2D
    
    h_layer.update_para(del_weight,del_bias_a,del_bias_b, lambda );
    
    return 0;

    
}
int DRM::update_RBM_1D( int d, std::vector< std::vector< std::complex<double> > >  &del_weight, std::vector<std::complex<double> >  &del_bias_b, double lambda )
{
    d_vector[d].update_para( del_weight , del_bias_b, lambda);
    
    return 0;
}











//--------------------------------sampling part ------------------------------
double DRM::random_num()
{// generate random number
    return 1.0*rand()/(RAND_MAX + 1.0);
}


int DRM::gibbs_sampling()
{
    int i,j,k;
    int dep;
    
    double accept;
    double r;
    
   
        
        // update x
        for(i=0;i<N_visible;i++)
        {
            for(j=0;j<N_visible;j++)
            {
                accept = sigmoid_n(i % N_block, j % N_block);
                
                r = random_num();
                
                
                if( (r< accept && state_1[i][j]==0) || (r>=accept && state_1[i][j]==1 ) )
                {
                    update_n( i, j);
                }
                
            }
        }

        //update h
        for(dep=0; dep< int(N_hidden.size()) ;dep++)
        {
            for(k=0; k< N_hidden[dep] ;k++)
            {
                
                accept = sigmoid_h(dep, k);
                r = random_num();
                
               
                
                if( (r< accept && h_state_1[dep][k]==0) || (r>=accept && h_state_1[dep][k]==1 ) )
                {
                    update_h( dep, k);
                }
                
            }
            
            
        }
        
   
    
    return 1;
    
    
    
}

//-----------------------set random()_--------------------

int DRM::para_random()
{
    
    double re,im;
    std::complex<double> unit_i(0.0,1.0);
    
    
    int i,j,k,m,dep;
    
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block;j++)
        {
            for(k=0;k<N_hidden[0];k++)
            {
                re = random_num()-0.5;
                im = random_num()-0.5;
                
        
                
                h_layer.weight[i][j][k] = (re+im*unit_i) * double( N_visible );
                
                
            }
            re = random_num()-0.5;
            im = random_num()-0.5;
            
            
            h_layer.bias_a[i][j] = ( re + im * unit_i ) * double (N_visible );
            
        }
    }
    
    for(k=0;k<N_hidden[0];k++)
    {
        re = random_num()-0.5;
        im = random_num()-0.5;
        
        
        h_layer.bias_b[k] = re+im*unit_i;
    }
    
    for(dep = 0 ; dep < int(N_hidden.size()-1); dep++)
    {
        for( k = 0 ;k< N_hidden[dep+1] ; k++)
        {
            for(m=0;m<N_hidden[dep];m++)
            {
                re = random_num()-0.5;
                im = random_num()-0.5;
                d_vector[dep].weight[m][k] = re+im*unit_i;;
                
            }
            re = random_num()-0.5;
            im = random_num()-0.5;
            d_vector[dep].bias_b[k] =re+im*unit_i;
        }
    }
    
    
    
    return 0;
}
int DRM::state_random()
{
    
    double p;
    
    
    int i,j,k,m;
    
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block;j++)
        {
        
            p=random_num();
            
            if(p>0.5)
            {
                state_1[i][j]=1;
            }
            else
            {
                state_1[i][j]=0;
            }
            
           
          
            
        }
    }

    
    for(k = 0 ; k < int(N_hidden.size()); k++)
    {
        for(m=0;m<N_hidden[k];m++)
        {
            p=random_num();
            
            if(p>0.5)
            {
                h_state_1[k][m]=1;
            }
            else
            {
                h_state_1[k][m]=0;
            }
            
            p=random_num();
            
        }
    }
    
    
    
    return 0;
}



int DRM::para_const(std::complex<double> constant)
{
    

  
    
    
    int i,j,k,m,dep;
    
   
    for(i=0;i<N_block;i++)
    {
        for(j=0;j<N_block;j++)
        {
            for(k=0;k<N_hidden[0];k++)
            {
               
                
                
                h_layer.weight[i][j][k] = constant* double(N_visible) ;
                
                
            }
           
            
            h_layer.bias_a[i][j] = constant *double(N_visible) ;
            
        }
    }
   
    
    for(k=0;k<N_hidden[0];k++)
    {
       
        
        h_layer.bias_b[k] =constant;
    }
    
    
    for(dep = 0 ; dep < int(N_hidden.size()-1); dep++)
    {
        for( k = 0 ;k< N_hidden[dep+1] ; k++)
        {
            for(m=0;m<N_hidden[dep];m++)
            {
                
                d_vector[dep].weight[m][k] = constant;;
                
            }
            
            d_vector[dep].bias_b[k] =constant;
        }
    }
    
    
    
    return 0;
}










//-----------------------------output------------------

int DRM::write_network( std::string filename)
{
    std::fstream file;
    

    file.open("network.txt", std::ios::out);
    
    
    
    if( !file.is_open() )
    {
        std::cout<<"can't open network output file"<<std::endl;
        exit(2);
    }
    
    
    
    int i,j,k;
    file<<"weight 2d\n";
    for(i=0;i<N_block ; i++)
    {
        for(j=0;j<N_block ; j++)
        {
            
            for(k=0;k<N_hidden[0]; k++)
            {
                file<<h_layer.weight[i][j][k].real()<<" " <<h_layer.weight[i][j][k].imag()<<"  ";
            }
            file<<"\n";
        }
    }
    file<<"bias a 2d\n";
    for(i=0;i<N_block ; i++)
    {
        for(j=0;j<N_block ; j++)
        {
            file<<h_layer.bias_a[i][j].real()<<" " <<h_layer.bias_a[i][j].imag()<<"  ";
            
            
        }
        file<<"\n";
    }
    
    file<<"bias b 2d\n";
    for(k=0;k<N_hidden[0];k++)
    {
        file<<h_layer.bias_b[k].real()<<" "<<h_layer.bias_b[k].imag()<<"  ";
    }
    file<<"\n";
    
    
    file<<"1d layer\n";
    
    for( int d=0; d< int (d_vector.size() ) ;d++)
    {
        
        file<<"-----------------layer " << d+1<<"--------------------\n";
        file<<"weight \n";
        for(i=0; i<N_hidden[d] ; i++)
        {
            for(j=0;j<N_hidden[d+1];j++)
            {
                file<<d_vector[d].weight[i][j].real()<<" "<<d_vector[d].weight[i][j].imag()<<"  ";
            }
            file<<"\n";
        }
        file<<"bias b \n";
        
        for(i=0;i<N_hidden[d+1];i++)
        {
            file<<d_vector[d].bias_b[i].real()<<" "<<d_vector[d].bias_b[i].imag()<<"  ";
        }
        file<<"\n";
        
        
    }
    
    
    
    file.close();
    
    return 1;
    
    
}

























//-----------------------TEST-------------------------------


std::complex<double> DRM::return_p( std::vector< std::vector <int> > &state, std::vector< std::vector <int> > &h_state )
{
    int i,j,k;
    int layer;
    int index;
    std::complex<double> E=0.0;

    for(i=0;i<N_visible; i++)
    {
        for(j=0;j<N_visible; j++)
        {
            E = E+ h_layer.bias_a[i%N_block][j%N_block] * double( state[i][j] )/ double( N_visible * N_visible);
            
            
            for(k=0;k<N_hidden[0];k++)
            {
                E = E + h_layer.weight[i%N_block][j%N_block][k] * double(state[i][j] * h_state[0][k] )/ double( N_visible * N_visible);
               
            }
            
        }
    }
    
    for(i=0;i<N_hidden[0];i++)
    {
        E=E + h_layer.bias_b[i] * double( h_state[0][i] );
      
    }
    
    
    for(layer = 1; layer < int(N_hidden.size()) ; layer++)
    {
        
        index = layer-1;
        
        for(i=0; i<N_hidden[layer-1] ; i++)
        {
            for(j=0; j<N_hidden[layer] ; j++)
            {
                E = E + d_vector[index].weight[i][j] * double( h_state[layer-1][i] * h_state[layer][j] );

            }
        }
        
        for(i=0 ; i<N_hidden[layer] ; i++)
        {
            E = E+ d_vector[index].bias_b[i] * double( h_state[layer][i] );
        }
        
        
        
    }

    return std::exp(-E) ;

    
    
}



std::complex<double> DRM::return_p_hid( std::vector< std::vector <int> > &h_state )
{
    
    
    int i,j;
    int layer;
    int index;
    std::complex<double> E=0.0;
    
    
    for(i=0;i<N_hidden[0];i++)
    {
        E=E + h_layer.bias_b[i] * double( h_state[0][i] );
        
    }
    
    
    for(layer = 1; layer < int(N_hidden.size()) ; layer++)
    {
        
        index = layer-1;
        
        for(i=0; i<N_hidden[layer-1] ; i++)
        {
            for(j=0; j<N_hidden[layer] ; j++)
            {
                E = E + d_vector[index].weight[i][j] * double( h_state[layer-1][i] * h_state[layer][j] );
                
            }
        }
        
        for(i=0 ; i<N_hidden[layer] ; i++)
        {
            E = E+ d_vector[index].bias_b[i] * double( h_state[layer][i] );
        }
        
        
        
    }
    
    return std::exp(-E) ;

    
    
    
    
    
    
    
    
}


std::vector< std::vector<int> > generate_n_drm(int n , int N_lattice)
{
    
    std::vector< std::vector<int> > re;
    
    
    
    
    int i,j;
    int indi,indj;
    int index=0;
    
    int rmd;
    re.resize(N_lattice);
    
    for( i=0 ;i <N_lattice ; i++)
    {
        re.resize(N_lattice);
    }
    
    for(i=0;i<N_lattice ;i++)
    {
        for(j=0;j<N_lattice ;j ++ )
        {
            re[i][j]=0;
        }
    }
    
    while(1)
    {
        rmd = n%2;
        
 
        
        indj = index%N_lattice;
        
        indi = (index-indj)/N_lattice;
        
        re[indi][indj]=rmd;
        index=index+1;
        
        n=(n-rmd)/2;
        if(n==0)
        {
            break;
        }
        
    }
    return re;
    
}


    
    
    
    
    
    
    
    
    
    
    

