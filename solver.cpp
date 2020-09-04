//
//  solver.cpp
//  neural_network_state
//
//  Created by huhaoyu on 2018/11/13.
//  Copyright © 2018 huhaoyu. All rights reserved.
//

#include "solver.hpp"
/*
solver::solver( std::string input_file,int N_block, int N_visible, std::vector<int> N_hidden, std::vector<std::string> parameter_file, std::vector<std::string> state_file, int p_minimum_in):network(N_block,N_visible,N_hidden,parameter_file,state_file), obs( "ising" , N_block, N_visible, N_hidden, p_minimum_in )
{
    std::fstream file;
    
    file.open( input_file , std::ios::in);
    
    if( !file.is_open())
    {
        std::cout<<"no input file "<<std::endl;
        exit(4);
    }
    
    p_minimum = p_minimum_in;
    
    file>>lambda;
    
    file>>freq_measure>>n_sample>>n_sweep>>n_warmup;
    
    file.close();
    
    
    
    
}

 */

solver::solver(int N_block, int N_visible, std::vector<int> N_hidden, double p_minimum_in):network(N_block,N_visible,N_hidden), obs( "ising" , N_block, N_visible, N_hidden, p_minimum_in )
{
    std::fstream file;
    double h,J;
    
    file.open( "input.txt" , std::ios::in);
    
    if( !file.is_open())
    {
        std::cout<<"no input file "<<std::endl;
        exit(4);
    }
    
    p_minimum = p_minimum_in;
    
    file>>lambda;
    
    file>>freq_measure>>n_sample>>n_sweep>>n_warmup;
    
    file>>h>>J;
    
    obs.Hamiltonian.h=h;
    obs.Hamiltonian.Jz=J;
    file.close();
    
    
    
    
    
}


int solver::solve()
{
    
    
    int i=0,j = 0;

    
    network.para_random();
    network.state_random();
    
    
    for(i=0;i<n_sweep;i++)
    {
        std::cout<<"sweep"<<" "<<i<<std::endl;
        std::cout<<"Monte Carlo Sampling..."<<std::endl;
        obs.set_zero();
        network.initialize();
        
         for(j=0;j<n_warmup;j++)
         {
         network.gibbs_sampling();
         //network.gibbs_sampling(2);
         
         
         }
        
        for(j=0;j<n_sample;j++)
        {
            network.gibbs_sampling();
            //network.gibbs_sampling(2);
            
          
            
            
            if(j%1 == 0)
            {
            
                //obs.get_observable(network.state_1, network.h_state_1, network.state_2, network.h_state_2, network.p_1, network.p_2);
                
                obs.get_observable(network.state_1, network.h_state_1, network.p_1, network);
            }
            
            
              //std::cout<<"h,i "<<obs.Htilde<<" "<<obs.Itilde<<"\n";
           
            
        }
        
        
        
        
        if( obs.get_average() == 0 )
        {
            continue ;
        }
        
        
        
        
        obs.cal_partial();
        
        std::cout<<"Energy="<<obs.Energy<<" Mag="<<obs.magnetization<<"\n";
        write_energy(i);
        write_network( "network.txt" );
        //cal_en_3lay();
        
        
        
        
        network.update_RBM_2D(obs.partial_weight_2D, obs.partial_bias_a_2D, obs.partial_bias_b_2D, lambda);
        
        
         for(int d=0; d< int( network.d_vector.size()); d++)
         {
             network.update_RBM_1D( d, obs.partial_weight_1D[d] , obs.partial_bias_b_1D[d], lambda );
         }
        
        
    }
    
    
    
    
    
    
    
    return 1;
    
    
    
}


int solver::write_energy(int sweep)
{
    
    std::ofstream file;
    
    
    
    file.open("energy.txt", std::ios::app);
    
    double ee;
    
    ee = get_entangelment_entropy(network, network.N_visible, network.N_block, network.N_hidden);
    
    
    
    if( !file.is_open() )
    {
        std::cout<<"can't open energy.txt"<<std::endl;
        exit(2);
    }
    
    file<<sweep<<" "<<obs.Energy<< "  "<<obs.magnetization.real()<<" "<<ee<<std::endl;
    
    
    
    
    file.close();
    
    return 1;
    
}

int solver::write_network( std::string filename)
{
    network.write_network(filename);
    
    return 0;
    
}





//----------test-------------------




std::vector< std::vector<int> > generate_n(int n , int N_lattice)
{
    
    std::vector< std::vector<int> > re;
    
    
    
    int i,j;
    int indi,indj;
    int index=0;
    
    int rmd;

    re.resize(N_lattice);
    for( i=0; i<N_lattice; i++)
    {
        re[i].resize(N_lattice,0);
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

double solver::cal_en_3lay()
{
    int i,j,k,m;
    int k2,m2;
    

    
    int x,y,z;


    
    
    std::vector< std::vector<int> > n1,n2;
    
    std::complex<double> H, h12;
    std::complex<double> I, i12;
    std::complex<double> Z;
    
    std::complex<double> p1;
    std::complex<double> p2;
    
    std::vector< std::vector< int > > h1,h2;
    
    std::vector< std::vector< std::vector< std::complex<double> > > > o_weight_0,i_weight_0;
    
    std::complex<double> o_weight_1, i_weight_1;
    
    std::complex<double> o_bias_b_1, i_bias_b_1, o_bias_b_0, i_bias_b_0;
    
    std::vector< std::vector< std::complex<double> > > o_bias_a, i_bias_a;
    
    
    o_weight_0.resize(network.N_block,  std::vector< std::vector< std::complex<double> > >(network.N_block, std::vector< std::complex<double> >( network.N_hidden[0],0.0)  )   );
    i_weight_0.resize(network.N_block,  std::vector< std::vector< std::complex<double> > >(network.N_block, std::vector< std::complex<double> >( network.N_hidden[0],0.0)  )   );
    
    o_weight_1=0.0;
    i_weight_1=0.0;
    o_bias_b_1=0.0;
    i_bias_b_1=0.0;
    
    o_bias_b_0=0.0;
    i_bias_b_0=0.0;
    
    o_bias_a.resize( network.N_block, std::vector< std::complex<double> > ( network.N_block, 0.0 ) );
    i_bias_a.resize( network.N_block, std::vector< std::complex<double> > ( network.N_block, 0.0 ) );
    
    
    for(i=0;i<network.N_block;i++)
    {
        for(j=0;j<network.N_block;j++)
        {
            for(k=0;k<network.N_hidden[0];k++)
            {
                
                o_weight_0[i][j][k]=0.0;
                i_weight_0[i][j][k]=0.0;
                
                
            }
            o_bias_a[i][j]=0.0;
            i_bias_a[i][j]=0.0;
            
        }
    }
    
    
    
    h1.resize(2, std::vector<int> (1) );
    h2.resize(2, std::vector<int> (1) );
    
    H=0.0;
    I=0.0;
    Z=0.0;
    for(i=0;i<pow(2,network.N_visible*network.N_visible ) ; i++)
    {

        n1 = generate_n(i, network.N_visible);
       
        
        
        for(j=0; j<pow(2,network.N_visible*network.N_visible ) ; j++ )
        {
            n2= generate_n(j, network.N_visible);
            
        
            for(k=0;k<2;k++)
            {
                h1[0][0]=k;
                for(m=0;m<2;m++)
                {
                    h2[0][0]=m;
                    
                    
                    for( k2=0;k2<2;k2++)
                    {
                        h1[1][0]=k2;
                        for( m2 =0; m2<2;m2++)
                        {
                            h2[1][0]=m2;
                            i12=1.0;
                            h12 = obs.Hamiltonian.get_element(n1, n2);
                            
                           
                                for(x=0;x<network.N_visible;x++)
                                {
                                    
                                    for(y=0;y<network.N_visible;y++)
                                    {
                                        if(n1[x][y] != n2[x][y])
                                        {
                                            i12=0.0;
                                            
                                        }
                                    }
                                }
                            
                            p1 = network.return_p(n1, h1);
                            p2 = network.return_p(n2, h2);
                            
                            
                            for(x=0;x<network.N_visible;x++)
                            {
                                for(y=0;y<network.N_visible;y++)
                                {
                                    for(z=0;z<network.N_hidden[0] ;z++)
                                    {
                                        o_weight_0[x%network.N_block][y%network.N_block][z] = o_weight_0[x%network.N_block][y%network.N_block][z]+std::conj(p1)*p2*(- double(n2[x][y] * h2[0][z]) )*h12/ double( network.N_visible * network.N_visible);
                
                                  
                                        i_weight_0[x%network.N_block][y%network.N_block][z] = i_weight_0[x%network.N_block][y%network.N_block][z]+std::conj(p1)*p2*(- double(n2[x][y] * h2[0][z]) )*i12/ double( network.N_visible * network.N_visible);

                                    }
                                    o_bias_a[x%network.N_block][y%network.N_block]=o_bias_a[x%network.N_block][y%network.N_block] + double(-n2[x][y])*h12*std::conj(p1)*p2/ double( network.N_visible * network.N_visible);
                                    i_bias_a[x%network.N_block][y%network.N_block]=i_bias_a[x%network.N_block][y%network.N_block] + double(-n2[x][y])*i12*std::conj(p1)*p2/ double( network.N_visible * network.N_visible);
                                }
                            }
           
                            o_bias_b_0=o_bias_b_0+double(-h2[0][0])*h12*std::conj(p1)*p2;
                            i_bias_b_0=i_bias_b_0+double(-h2[0][0])*i12*std::conj(p1)*p2;
                            
                            o_bias_b_1=o_bias_b_1+double(-h2[1][0])*h12*std::conj(p1)*p2;
                            i_bias_b_1=i_bias_b_1+double(-h2[1][0])*i12*std::conj(p1)*p2;
                            
                            o_weight_1=o_weight_1+double(-h2[1][0]*h2[0][0])*h12*std::conj(p1)*p2;
                            i_weight_1=i_weight_1+double(-h2[1][0]*h2[0][0])*i12*std::conj(p1)*p2;
                            
                            
                            
                            H=H+h12*p2*std::conj(p1);
                            I=I+i12*p2*std::conj(p1);
                            Z= Z+ p1*std::conj(p1);
                            
       
                        }
                        
                    }
                    
                    
                    
                
                    
                    
                }
            }
        }
    }



    Z=Z/std::pow(2.0, network.N_visible * network.N_visible + 2);
    std::cout<<"eaxt re"<<H/I<<" "<<I/Z<<std::endl;
    
    
    
    std::complex<double> tmp;
    for(x=0;x<network.N_block;x++)
    {
        for(y=0;y<network.N_block;y++)
        {
            for(z=0;z<network.N_hidden[0] ;z++)
            {
              
                
                tmp = o_weight_0[x%network.N_block][y%network.N_block][z]*I - i_weight_0[x%network.N_block][y%network.N_block][z]*H;
                
                tmp = tmp/I/I;
                std::cout<<tmp<<std::endl;
                
                //std::cout<<std::abs( tmp- obs.partial_weight_2D[x%network.N_block][y%network.N_block][z])/std::abs(tmp)  <<std::endl;
                
                //std::cout<<abs( o_weight_0[x][y][z]/Z - obs.O_D_weight_2D[x][y][z])/abs( o_weight_0[x][y][z] /Z)<<" "<<abs( i_weight_0[x][y][z]/Z - obs.I_D_weight_2D[x][y][z])/abs( i_weight_0[x][y][z] /Z)<<std::endl;
                
            }
        }
    }
    
    
    std::cout<<"BIAS_A"<<std::endl;
    for(x=0;x<network.N_block;x++)
    {
        for(y=0;y<network.N_block;y++)
        {
            
            tmp =o_bias_a[x%network.N_block][y%network.N_block]*I - i_bias_a[x%network.N_block][y%network.N_block]*H;
            
            tmp = tmp/I/I;
            
            std::cout<<tmp<<std::endl;
            
           // std::cout<<std::abs( tmp - obs.partial_bias_a_2D[x][y])/std::abs(tmp)<<std::endl;
                
           // std::cout<<abs( o_bias_a[x][y]/Z - obs.O_D_bias_a_2D[x][y])/abs(o_bias_a[x][y]/Z)<<" "<<abs( i_bias_a[x][y]/Z - obs.I_D_bias_a_2D[x][y])/abs(  i_bias_a[x][y]/Z)<<std::endl;
        }
    }

    tmp = (o_bias_b_0 * I - i_bias_b_0 * H)/(I*I);
    std::cout<<"o/i bias_b_0"<<" "<<std::abs( tmp - obs.partial_bias_b_2D[0] )/std::abs(tmp)<<std::endl;
    
    std::cout<<"o/i bias_b_0"<<" "<<std::abs( o_bias_b_0/Z- obs.O_D_bias_b_2D[0] )/std::abs( o_bias_b_0/Z)<<" "<< std::abs( i_bias_b_0/Z- obs.I_D_bias_b_2D[0] )/std::abs( i_bias_b_0/Z)<<std::endl;
    
    tmp = (o_bias_b_1 * I - i_bias_b_1 * H)/(I*I);
    std::cout<<"o/i bias_b_1"<<" "<< std::abs( tmp - obs.partial_bias_b_1D[0][0])/std::abs(tmp) << " "<< tmp<< " "<<obs.partial_bias_b_1D[0][0]<<std::endl;
    std::cout<<"o/i bias_b_1"<<" "<< std::abs( o_bias_b_1/Z  - obs.O_D_bias_b_1D[0][0])/std::abs(o_bias_b_1/Z )<<" "<<std::abs( i_bias_b_1/Z  - obs.I_D_bias_b_1D[0][0])/std::abs(i_bias_b_1/Z ) <<std::endl;
    
    
    
    tmp = (o_weight_1 * I - i_weight_1 * H)/(I*I);
    std::cout<<"o/i weight_1"<<" "<<std::abs( tmp - obs.partial_weight_1D[0][0][0] )/std::abs(tmp)<<std::endl;
     std::cout<<"o/i weight_1"<<" "<<std::abs( o_weight_1/Z - obs.O_D_weight_1D[0][0][0] )/std::abs(o_weight_1/Z )<<" "<< std::abs( i_weight_1/Z - obs.I_D_weight_1D[0][0][0] )/std::abs(i_weight_1/Z )<<std::endl;

    
    return 0;
    
}


