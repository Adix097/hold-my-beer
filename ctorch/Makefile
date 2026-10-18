CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -g -O0 -fsanitize=address,undefined -Iinclude
SRC = $(wildcard src/*.c)
TESTS = $(patsubst tests/%.c,build/%, $(wildcard tests/*.c))
TEST ?=
RUN_TESTS = $(if $(TEST),build/$(TEST),$(TESTS))

build/%: tests/%.c $(SRC) include/ctorch.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SRC) $< -o $@ -lm

test: $(RUN_TESTS)
	@for t in $(RUN_TESTS); do \
		echo "== $$t =="; \
		$$t || exit 1; \
	done

clean:
	rm -rf build

.PHONY: test clean