OBJS = DRM.o Ham.o main.o Observable.o RBM_1D.o RBM_2D.o solver.o entanglement_entropy.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(BEBUG)

main : $(OBJS)
	$(CC) -o main $(OBJS) 

%.o : %.cpp
	$(CC) $(CFLAGS) $<
clean:
	\rm *.o *~main

