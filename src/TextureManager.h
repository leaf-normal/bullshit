// TextureManager.h 修改
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
    std::vector<unsigned char> data;      // 用于LDR纹理
    std::vector<float> hdr_data;          // 用于HDR纹理
    std::shared_ptr<grassland::graphics::Image> gpuImage;
};

class TextureManager {
public:
    explicit TextureManager(grassland::graphics::Core* core) : core_(core) {}
    int LoadTexture(const std::string& filename);
    int LoadHDRTexture(const std::string& filename);  // 新增：加载HDR纹理
    Texture* GetTexture(int id);
    int TextureCount() const { return static_cast<int>(textures_.size()); }

private:
    grassland::graphics::Core* core_;
    std::vector<Texture> textures_;
};