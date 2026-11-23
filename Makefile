CC ?= gcc
SIZE ?= size
NM ?= nm
CFLAGS := -std=c11 -O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin
HOST_CFLAGS := -std=c11 -O3 -Wall -Wextra -Werror

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(OS),Windows_NT)
  SHARED_EXT := dll
  SHARED_FLAGS := -shared -nostdlib -Wl,-e,0 -Wl,--export-all-symbols
  EXE_EXT := .exe
else ifeq ($(UNAME_S),Darwin)
  SHARED_EXT := dylib
  SHARED_FLAGS := -dynamiclib -fPIC
  EXE_EXT :=
else
  SHARED_EXT := so
  SHARED_FLAGS := -shared -fPIC
  EXE_EXT :=
endif

.PHONY: all audit bench clean

all: csrc/edgesync.o csrc/libedgesync.$(SHARED_EXT) csrc/bench_edgesync$(EXE_EXT)

csrc/edgesync.o: csrc/edgesync.c csrc/edgesync.h
	$(CC) $(CFLAGS) -c csrc/edgesync.c -o csrc/edgesync.o

csrc/libedgesync.$(SHARED_EXT): csrc/edgesync.c csrc/edgesync.h
	$(CC) $(CFLAGS) $(SHARED_FLAGS) csrc/edgesync.c -o csrc/libedgesync.$(SHARED_EXT)

csrc/bench_edgesync$(EXE_EXT): csrc/bench_edgesync.c csrc/edgesync.o csrc/edgesync.h
	$(CC) $(HOST_CFLAGS) csrc/bench_edgesync.c csrc/edgesync.o -lm -o csrc/bench_edgesync$(EXE_EXT)

audit: csrc/edgesync.o
	$(SIZE) csrc/edgesync.o
	$(NM) -u csrc/edgesync.o

bench: csrc/bench_edgesync$(EXE_EXT)
	./csrc/bench_edgesync$(EXE_EXT)

clean:
	rm -f csrc/edgesync.o csrc/libedgesync.dll csrc/libedgesync.so csrc/libedgesync.dylib csrc/bench_edgesync csrc/bench_edgesync.exe
