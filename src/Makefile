CC	=	g++

CPPFLAGS	= -std=c++17
CLAGS	=	-g3	-Wall	-Wextra
LDFLAGS	=

OBJS	=graph.o	utils.o	test.o
PROJECT	=test

RM	=	rm	-rf

all:	$(PROJECT)
.PHONY:	all	clean

clean:
	$(RM)	$(OBJS)	$(PROJECT)

$(PROJECT):	$(OBJS)
	$(CC)	-o	$@	$^	$(LDFLAGS)

%.o:	%.cpp
	$(CC)	$(CPPFLAGS)	$(CFLAGS)	-o	$@	-c	$<

utils.o:	utils.hpp
graph.o:	graph.hpp
test.o:	utils.hpp	graph.hpp
