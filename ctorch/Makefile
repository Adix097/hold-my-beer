CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0 -fsanitize=address,undefined -Iinclude
SRC = $(wildcard src/*.c)

build/test_tensor: $(SRC) tests/test_tensor.c include/ctorch.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SRC) tests/test_tensor.c -o $@ -lm

test: build/test_tensor
	./build/test_tensor

clean:
	rm -rf build

.PHONY: test clean