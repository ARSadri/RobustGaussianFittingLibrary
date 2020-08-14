#ar -r RGFLib.a RGFLib.o
CC = gcc
SL = ar
SLFLAGS = -r
CFLAGS = -fPIC -Wall -O2 #-g
LDFLAGS = -shared
RM = rm -rf
TARGET_LIB = RGFLib.so

SRCS = RGFLib.c
OBJS = $(SRCS:.c=.o)
SLFILE = $(SRCS:.c=.a)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^
	$(SL) $(SLFLAGS) $(SLFILE) $(OBJS)
	
$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: test
test:
	python3 tests.py

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)
