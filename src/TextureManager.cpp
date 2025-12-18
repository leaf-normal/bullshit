#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "TextureManager.h"
#include <stdexcept>
#include "grassland/graphics/image.h"

int TextureManager::LoadTexture(const std::string& filename) {
    Texture tex{};
    unsigned char* img = stbi_load(filename.c_str(), &tex.width, &tex.height, &tex.channels, 4);
    if (!img) {
        throw std::runtime_error("Failed to load texture: " + filename);
    }
    tex.data.assign(img, img + tex.width * tex.height * 4);
    stbi_image_free(img);

    // 创建 GPU Image
    core_->CreateImage(tex.width, tex.height,
        grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM,
        &tex.gpuImage);

    tex.gpuImage->UploadData(tex.data.data());
    grassland::LogInfo("Texture loaded: {} ({}x{}, {} channels)", filename, tex.width, tex.height, tex.channels);
    textures_.push_back(std::move(tex));
    
    return static_cast<int>(textures_.size() - 1); // 返回纹理ID
}

Texture* TextureManager::GetTexture(int id) {
    if (id < 0 || id >= (int)textures_.size()) return nullptr;
    return &textures_[id];
}