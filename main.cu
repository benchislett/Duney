#include "lodepng.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

template<typename T>
class Grid {
public:
    enum State {Host, GPU} state;
    unsigned int width, height;

    __host__ Grid(unsigned int w, unsigned int h) : width(w), height(h), state(Host) {
        cudaMalloc(&vals_device, size());
        vals_host = (T*) calloc(length(), sizeof(T));
    }

    __host__ __device__ Grid(T* vh, T* vd, unsigned int w, unsigned int h, State st = Host) : vals_host(vh), vals_device(vd), width(w), height(h), state(st) {}

    __host__ __device__ ~Grid() {
#ifndef __CUDA_ARCH__
        free(vals_host);
        cudaFree(vals_device);
#endif
    }

    __host__ __device__ T& operator[](int index) {
#ifdef __CUDA_ARCH__
        assert (state == GPU);
        return vals_device[index];
#else
        assert (state == Host);
        return vals_host[index];
#endif
    }

    __host__ __device__ const T& operator[](int index) const {
#ifdef __CUDA_ARCH__
        assert (state == GPU);
        return vals_device[index];
#else
        assert (state == Host);
        return vals_host[index];
#endif
    } 

    __host__ __device__ T& at(int index_row, int index_column) {
        return this[index_row * width + index_column];
    }

    __host__ __device__ T& at(int index_row, int index_column) const {
        return this[index_row * width + index_column];
    }

    __host__ __device__ unsigned int length() const {
        return width * height;
    }

    __host__ __device__ unsigned int size() const {
        return width * height * sizeof(T);
    }

    __host__ T* begin() const {
        assert (state == Host);
        return vals_host;
    }

    __host__ T* end() const {
        assert (state == Host);
        return vals_host + (width * height);
    }

private:
    T *vals_host;
    T *vals_device;
};

unsigned int rescale(unsigned int val, unsigned int data_max, unsigned int new_max) {
    float scaled = (float)val * (float)new_max / (float)data_max;
    return (unsigned int)scaled;
}

void serialize(const char *filename, const Grid<unsigned int>& data)
{
    std::vector<unsigned char> raw_data;
    raw_data.reserve(data.width * data.height * 4);

    unsigned int max = *std::max_element(data.begin(), data.end());
    for (int i = 0; i < data.width * data.height; i++) {
        unsigned char byte = rescale(data[i], max, 255);
        for (int channel = 0; channel < 3; channel++)
            raw_data.push_back(byte);
        raw_data.push_back(255);
    }

    unsigned error = lodepng::encode(filename, raw_data, data.width, data.height);
    if (error)
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

int main()
{
    Grid<unsigned int> h(10, 10);

    serialize("tmp.png", h);
    return 0;
}
