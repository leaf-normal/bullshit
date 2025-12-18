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

int TextureManager::LoadHDRTexture(const std::string& filename) {
    Texture tex{};
    
    // stbi_set_flip_vertically_on_load(true); // HDR纹理通常需要垂直翻转
    
    // 加载HDR图像
    float* img = stbi_loadf(filename.c_str(), &tex.width, &tex.height, &tex.channels, 0);
    if (!img) {
        throw std::runtime_error("Failed to load HDR texture: " + filename);
    }
    
    size_t pixel_count = tex.width * tex.height;
    tex.hdr_data.resize(pixel_count * 4);
    
    if (tex.channels == 3) {
        // RGB -> RGBA
        for (size_t i = 0; i < pixel_count; ++i) {
            tex.hdr_data[i*4 + 0] = img[i*3 + 0];
            tex.hdr_data[i*4 + 1] = img[i*3 + 1];
            tex.hdr_data[i*4 + 2] = img[i*3 + 2];
            tex.hdr_data[i*4 + 3] = 1.0f; 
        }
    } else if (tex.channels == 4) {
        memcpy(tex.hdr_data.data(), img, pixel_count * 4 * sizeof(float));
    }
    
    stbi_image_free(img);
    
    core_->CreateImage(tex.width, tex.height,
        grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
        &tex.gpuImage);
    
    tex.gpuImage->UploadData(tex.hdr_data.data());
    
    grassland::LogInfo("HDR Texture loaded: {} ({}x{}, {} channels)", 
                      filename, tex.width, tex.height, tex.channels);
    
    textures_.push_back(std::move(tex));
    return static_cast<int>(textures_.size() - 1);
}

Texture* TextureManager::GetTexture(int id) {
    if (id < 0 || id >= (int)textures_.size()) return nullptr;
    return &textures_[id];
}