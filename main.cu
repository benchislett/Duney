#include "lodepng.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <random>
#include <chrono>


class ScopedTimer {
 public:
  ScopedTimer(std::string name) : m_name(std::move(name)),
        m_beg(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() / 1000.0f << " ms\n";
  }

 private:
  std::string m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

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

    __host__ __device__ const T& operator[](int index) const {
#ifdef __CUDA_ARCH__
        assert (state == GPU);
        return vals_device[index];
#else
        assert (state == Host);
        return vals_host[index];
#endif
    } 

    __host__ __device__ T& operator[](int index) {
        return const_cast<T &>(static_cast<const Grid<T> &>(*this)[index]);
    }

    __host__ __device__ const T& at(int index_row, int index_column) const {
        return operator[](index_row * width + index_column);
    }

    __host__ __device__ T& at(int index_row, int index_column) {
        return const_cast<T &>(static_cast<const Grid<T> &>(*this).at(index_row, index_column));
    }

    __host__ __device__ const T& at_wrap(int index_row, int index_column) const {
        // TODO: use power-of-two optimization
        return at((index_row + height) % height, (index_column + width) % width);
    }

    __host__ __device__ T& at_wrap(int index_row, int index_column) {
        return const_cast<T &>(static_cast<const Grid<T> &>(*this).at_wrap(index_row, index_column));
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
    int length = 512;
    int width = length, height = length;
    Grid<unsigned int> hmap(width, height);

    int min_height = 1;
    int max_height = 3;
    std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr(min_height, max_height); // define the (inclusive) range

    {
        ScopedTimer _timer("Height Map");
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                hmap.at(row, col) = distr(gen);
            }
        }
    }

    Grid<unsigned int> shadow(width, height);

    {
        ScopedTimer _timer("Shadow Map");
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                bool in_shadow = false;

                float curr_height = (float) hmap.at(row, col);
                float curr_x = (float) col + 0.5f; // check from center for 50% coverage

                for (int prev_col = col - 1; prev_col >= 0; prev_col--) {
                    float prev_height = (float) hmap.at(row, prev_col);
                    float prev_x = (float) prev_col + 1.0f; // check the rightmost point as it casts the furthest shadow

                    float rise = (prev_height - curr_height) / 3.0f; // slabs are 1/3 units tall
                    if (rise <= 0) continue;
                    float run = curr_x - prev_x;
                    float ratio = rise / run;
                    if (ratio > 0.26795) {
                        in_shadow = true;
                        break;
                    }

                    // early stoppage check based on global max height
                    ratio = ((float)max_height - curr_height) / (3.0f * run);
                    if (ratio < 0.26795) {
                        break;
                    }
                }

                shadow.at(row, col) = in_shadow ? 1 : 0;
            }
        }
    }
    
    {
        ScopedTimer _timer("Output");
        serialize("heightmap.png", hmap);
        serialize("shadowmap.png", shadow);
    }
    return 0;
}
