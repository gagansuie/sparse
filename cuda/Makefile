# Tenpak CUDA Kernels Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -shared -Xcompiler -fPIC
NVCC_FLAGS += -gencode arch=compute_70,code=sm_70  # V100
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75  # T4
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80  # A100
NVCC_FLAGS += -gencode arch=compute_86,code=sm_86  # RTX 3090
NVCC_FLAGS += -gencode arch=compute_89,code=sm_89  # RTX 4090
NVCC_FLAGS += -gencode arch=compute_90,code=sm_90  # H100

TARGET = libtenpak_cuda.so
SRC = awq_gemm.cu

.PHONY: all clean test benchmark

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 tenpak_cuda.py --test

benchmark: $(TARGET)
	python3 -c "from tenpak_cuda import benchmark_g8; benchmark_g8()"

# Install to system
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	ldconfig

# Development build with debug info
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)
