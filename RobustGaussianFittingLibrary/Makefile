CC = gcc
#CFLAGS = -fPIC -Wall -g
CFLAGS = -fPIC -O2
LDFLAGS = -shared
RM = rm -rf
TARGET_LIB = RGFLib.so

SRCS = RGFLib.c
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: test
test:
	python3 tests.py

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.c=.d)
