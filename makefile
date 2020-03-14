OBJS = dmrgchain-dangerous.o MPSObjects.o lalgebra.o
CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG) -larmadillo
LFLAGS = -larmadillo

dmrgchain : $(OBJS)
	$(CC) -o dmrgchain-dangerous $(OBJS) $(LFLAGS)

dmrgchain.o : dmrgchain-dangerous.cpp MPSObjects.h lalgebra.h
	$(CC) $(CFLAGS) dmrgchain-dangerous.cpp

MPSObjects.o : MPSObjects.cpp MPSObjects.h
	$(CC) $(CFLAGS) MPSObjects.cpp

lalgebra.o : lalgebra.h lalgebra.cpp
	$(CC) $(CFLAGS) lalgebra.cpp

clean:
	\rm *.o *~ dmrgchain-dangerous

tar:
	tar cfv p1.tar dmrgchain-dangerous.cpp MPSObjects.cpp MPSObjects.h lalgebra.h lalgebra.cpp
