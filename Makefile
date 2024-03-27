DIRS = lib main

.PHONY: all clean dep
.SUFFIXES : .c .o

all :
	@for d in $(DIRS); \
	do \
		make -C $$d; \
	done

clean :
	@for d in $(DIRS); \
	do \
		make -C $$d clean; \
	done

dep :
	@for d in $(DIRS); \
	do \
		make -C $$d dep; \
	done
