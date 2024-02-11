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

    __host__ __device__ int wrap_row(int row) const {
        return (row + height) % height;
    }

    __host__ __device__ int wrap_col(int col) const {
        return (col + width) % width;
    }

    __host__ __device__ const T& at_wrap(int index_row, int index_column) const {
        // TODO: use power-of-two optimization
        return at(wrap_row(index_row), wrap_col(index_column));
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

int rescale(int val, int data_max, unsigned int new_max) {
    float scaled = (float)val * (float)new_max / (float)data_max;
    return (unsigned int)scaled;
}

template<typename T>
void serialize(const char *filename, const Grid<T>& data)
{
    std::vector<unsigned char> raw_data;
    raw_data.reserve(data.width * data.height * 4);

    int max = *std::max_element(data.begin(), data.end());
    int min = *std::min_element(data.begin(), data.end());
    for (int i = 0; i < data.width * data.height; i++) {
        unsigned char byte = rescale(data[i] + min, min + max, 255);
        for (int channel = 0; channel < 3; channel++)
            raw_data.push_back(byte);
        raw_data.push_back(255);
    }

    unsigned error = lodepng::encode(filename, raw_data, data.width, data.height);
    if (error)
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

bool climb_next(Grid<unsigned int>& hmap, int row, int col, int& out_row, int& out_col) {
    int current = hmap.at_wrap(row, col);
    int vals[6] = {-1, 1, 0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
        int dx = vals[i + 0];
        int dy = vals[i + 2];

        int next = hmap.at_wrap(row + dy, col + dx);
        if (next - current > 2) {
            out_row = row + dy;
            out_col = col + dx;
            return true;
        }
    }
    out_row = row;
    out_col = col;
    return false;
}

void climb(Grid<unsigned int>& hmap, int row, int col, int& out_row, int& out_col) {
    while (climb_next(hmap, row, col, out_row, out_col)) {
        row = out_row;
        col = out_col;
    }
    out_row = hmap.wrap_row(out_row);
    out_col = hmap.wrap_row(out_col);
}

bool descend_next(Grid<unsigned int>& hmap, int row, int col, int& out_row, int& out_col) {
    int current = hmap.at_wrap(row, col);
    int vals[6] = {-1, 1, 0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
        int dx = vals[i + 0];
        int dy = vals[i + 2];

        int next = hmap.at_wrap(row + dy, col + dx);
        if (next - current < -2) {
            out_row = row + dy;
            out_col = col + dx;
            return true;
        }
    }
    out_row = row;
    out_col = col;
    return false;
}

void descend(Grid<unsigned int>& hmap, int row, int col, int& out_row, int& out_col) {
    while (descend_next(hmap, row, col, out_row, out_col)) {
        row = out_row;
        col = out_col;
    }
    out_row = hmap.wrap_row(out_row);
    out_col = hmap.wrap_row(out_col);
}

void compute_shadows(const Grid<unsigned int> &hmap, Grid<bool> &shadowmap) {
    int max = *std::max_element(hmap.begin(), hmap.end());

    for (int row = 0; row < hmap.height; row++) {
        for (int col = 0; col < hmap.width; col++) {
            bool in_shadow = false;

            float curr_height = (float) hmap.at(row, col);
            float curr_x = (float) col + 0.5f; // check from center for 50% coverage

            for (int prev_col = col - 1; ; prev_col--) {
                float prev_height = (float) hmap.at_wrap(row, prev_col);
                float prev_x = (float) prev_col + 1.0f; // check the rightmost point as it casts the furthest shadow

                float rise = (prev_height - curr_height) / 3.0f; // slabs are 1/3 units tall
                float run = curr_x - prev_x;
                float ratio = rise / run;
                if (rise > 0) {
                    if (ratio > 0.26795) {
                        in_shadow = true;
                        break;
                    }
                }

                // early stoppage check based on global max height
                ratio = ((float)max - curr_height) / (3.0f * run);
                if (ratio <= 0.26795) {
                    break;
                }
            }

            shadowmap.at(row, col) = in_shadow ? 1 : 0;
        }
    }
}

int main()
{
    int length = 256;
    int width = length, height = length;
    Grid<unsigned int> hmap(width, height);

    int min_height = 1;
    int max_height = 3;
    std::random_device rd;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distr_heightmap(min_height, max_height);
    std::uniform_int_distribution<> distr_rows(0, height - 1);
    std::uniform_int_distribution<> distr_cols(0, width - 1);
    std::uniform_real_distribution<float> distr_01(0.0f, 1.0f);

    {
        ScopedTimer _timer("Height Map");
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                hmap.at(row, col) = distr_heightmap(gen);
            }
        }
    }

    Grid<bool> shadow(width, height);

    {
        ScopedTimer _timer("Shadow Map");
        compute_shadows(hmap, shadow);
    }

    {
        ScopedTimer _timer("Update");
        for (int epoch = 0; epoch < 1000; epoch++) {

            for (int iter = 0; iter < width * height; iter++) {
                int row = distr_rows(gen);
                int col = distr_cols(gen);
                if (hmap.at(row, col) == 0) continue;
                if (shadow.at(row, col) == 1) continue;

                // remove the grain from this position
                int erode_row, erode_col;
                climb(hmap, row, col, erode_row, erode_col);
                hmap.at(erode_row, erode_col)--;

                // hop downwind
                int deposit_col = erode_col;
                while (true) {
                    deposit_col = hmap.wrap_col(deposit_col + 1); // saltation length
                    if (shadow.at(erode_row, deposit_col) == 1) {
                        descend(hmap, erode_row, deposit_col, erode_row, deposit_col);
                        hmap.at(erode_row, deposit_col)++;
                        break;
                    } else {
                        bool has_sand = hmap.at(erode_row, deposit_col) > 0;
                        if (distr_01(gen) < (has_sand ? 0.6f : 0.4f)) {
                            descend(hmap, erode_row, deposit_col, erode_row, deposit_col);
                            hmap.at(erode_row, deposit_col)++;
                            break;
                        }
                    }
                }
            }

            compute_shadows(hmap, shadow);

            std::string name = "outputs/" + std::to_string(epoch) + ".png";
            serialize(name.c_str(), hmap);
        }
    }
    
    {
        ScopedTimer _timer("Output");
        serialize("heightmap.png", hmap);
        serialize("shadowmap.png", shadow);
    }
    return 0;
}
