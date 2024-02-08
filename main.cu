#include "lodepng.h"

#include <iostream>
#include <vector>

void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height) {
    unsigned error = lodepng::encode(filename, image, width, height);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main() {
    std::vector<unsigned char> data = {255, 0, 255, 255, 0, 0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255};
    encodeOneStep("tmp.png", data, 2, 2);
    return 0;
}
