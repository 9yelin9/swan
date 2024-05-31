CC = gcc
AR = ar rv
RM = rm -rf

USR_DIR = /home/9yelin9/.local

CFLAGS = -g -O2 -Wall -mcmodel=medium -I../include -I$(USR_DIR)/include -fPIC -fopenmp
LDFLAGS = -shared -L$(USR_DIR)/lib -L$(USR_DIR)/lib64 -fopenmp 
LINKS = -lz -lm -lblas -llapack

OBJS = libsw.o
LIBS_A = libsw.a
LIBS_S = ../libsw.so

.PHONY: all clean dep
.SUFFIXES : .c .o

.c .o :
	$(CC) $(CFLAGS) -c $<

all : $(LIBS_A) $(LIBS_S)
clean :
	$(RM) *.o
	$(RM) $(LIBS_A) $(LIBS_S)
dep :
	$(CC) $(CFLAGS) -M $(OBJS:.o=.c) 

$(LIBS_A) : $(OBJS)
	$(AR) $@ $(OBJS)
$(LIBS_S) : $(OBJS)
	$(CC) $(LDLIBS) $(LDFLAGS) -o $(LIBS_S) $(OBJS) $(LINKS)
