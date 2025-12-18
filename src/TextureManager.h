#pragma once
#include <string>
#include <vector>
#include <memory>
#include "grassland/graphics/Core.h"
#include "grassland/graphics/Image.h"

struct Texture {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;
    std::shared_ptr<grassland::graphics::Image> gpuImage;
};

class TextureManager {
public:
    explicit TextureManager(grassland::graphics::Core* core) : core_(core) {}
    int LoadTexture(const std::string& filename);
    Texture* GetTexture(int id);
    int TextureCount() const { return static_cast<int>(textures_.size()); }

private:
    grassland::graphics::Core* core_;
    std::vector<Texture> textures_;
};
