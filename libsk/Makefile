CC = gcc
RM = rm -f 

USR_DIR = /home/9yelin9/.local
GSL_DIR = /opt/gsl/gcc-11.4.1/2.7.1
OPENBLAS_DIR = /opt/openblas/gcc-11.4.1/0.3.26

CFLAGS = -g -O2 -Wall -mcmodel=medium -fgnu89-inline -std=c99 -Wno-format-overflow -I$(USR_DIR)/include -I$(GSL_DIR)/include -I$(OPENBLAS_DIR)/include -fPIC -fopenmp
LDFLAGS = -L$(USR_DIR)/lib -L$(GSL_DIR)/lib -L$(OPENBLAS_DIR)/lib -fopenmp
LINKS = -lz -lm -lblas -llapack -lgsl -lopenblas -lhdf5

OBJS = t2.o matrix.o
LIBS = ../libsk.so

.PHONY : all clean  
.SUFFIXES : .c .o

.c .o :
	$(CC) $(CFLAGS) -o $@ $< 

all : $(LIBS)
clean :
	$(RM) *.o 
	$(RM) $(LIBS)

t2.o:	$($@:.o=.c)	lattice.h f2c.h matrix.h clapack.h Makefile 
matrix.o:	$($@:.o=.c)	matrix.h Makefile

$(LIBS) : $(OBJS)
	$(CC) $(LDLIBS) $(LDFLAGS) -o $@ $(OBJS) $(LINKS) -shared
