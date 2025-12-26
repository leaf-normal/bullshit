#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "TextureManager.h"
#include <stdexcept>
#include "grassland/graphics/image.h"

static int ComputeMaxMipLevels(int w, int h) {
    int levels = 1;
    while (w > 1 || h > 1) {
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
        levels++;
    }
    return levels;
}

// LDR: RGBA8
std::vector<std::vector<unsigned char>>
TextureManager::BuildMipChainRGBA8(const unsigned char* base, int width, int height) {
    std::vector<std::vector<unsigned char>> mips;
    mips.emplace_back(base, base + width * height * 4);
    int w = width, h = height;
    while (w > 1 || h > 1)
    {
        int nw = std::max(1, w / 2);
        int nh = std::max(1, h / 2);
        std::vector<unsigned char> next(nw * nh * 4);
        auto fetch = [&](int x, int y, int c) {
            int ix = std::clamp(x, 0, w - 1);
            int iy = std::clamp(y, 0, h - 1);
            return mips.back()[(iy * w + ix) * 4 + c];
        };
        for (int y = 0; y < nh; ++y) {
            for (int x = 0; x < nw; ++x) {
                int sx = x * 2, sy = y * 2;
                for (int c = 0; c < 4; ++c) {
                    int sum = 0;
                    sum += fetch(sx + 0, sy + 0, c);
                    sum += fetch(sx + 1, sy + 0, c);
                    sum += fetch(sx + 0, sy + 1, c);
                    sum += fetch(sx + 1, sy + 1, c);
                    next[(y * nw + x) * 4 + c] = (unsigned char)(sum / 4);
                }
            }
        }
        mips.push_back(std::move(next));
        w = nw; h = nh;
    }
    return mips;
}
//HDR: RGB32
std::vector<std::vector<float>>
TextureManager::BuildMipChainRGBA32(const float* base, int width, int height) {
    std::vector<std::vector<float>> mips;
    mips.emplace_back(base, base + width * height * 4);
    int w = width, h = height;
    while (w > 1 || h > 1) {
        int nw = std::max(1, w / 2);
        int nh = std::max(1, h / 2);
        std::vector<float> next(nw * nh * 4);
        auto fetch = [&](int x, int y, int c) {
            int ix = std::clamp(x, 0, w - 1);
            int iy = std::clamp(y, 0, h - 1);
            return mips.back()[(iy * w + ix) * 4 + c];
        };
        for (int y = 0; y < nh; ++y) {
            for (int x = 0; x < nw; ++x) {
                int sx = x * 2, sy = y * 2;
                for (int c = 0; c < 4; ++c) {
                    float sum = 0;
                    sum += fetch(sx + 0, sy + 0, c);
                    sum += fetch(sx + 1, sy + 0, c);
                    sum += fetch(sx + 0, sy + 1, c);
                    sum += fetch(sx + 1, sy + 1, c);
                    next[(y * nw + x) * 4 + c] = (float)(sum / 4.0);
                }
            }
        }
        mips.push_back(std::move(next));
        w = nw; h = nh;
    }
    return mips;
}

int TextureManager::LoadTexture(const std::string& filename) {
    Texture tex{};
    unsigned char* img = stbi_load(filename.c_str(), &tex.width, &tex.height, &tex.channels, 4);
    if (!img) {
        throw std::runtime_error("Failed to load texture: " + filename);
    }
    tex.data.assign(img, img + tex.width * tex.height * 4);
    stbi_image_free(img);

    auto mip_chain = BuildMipChainRGBA8(tex.data.data(), tex.width, tex.height);
    tex.mip_levels = (int)mip_chain.size();
    int w = tex.width, h = tex.height;
    tex.has_mipmap=(tex.mip_levels>1);
    for (int level = 0; level < tex.mip_levels; ++level) {
        std::shared_ptr<grassland::graphics::Image> image;
        core_->CreateImage(w, h, grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
        image->UploadData(mip_chain[level].data());
        tex.mipImages.push_back(image);
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
    }
    tex.gpuImage = tex.mipImages[0];
    grassland::LogInfo("Texture loaded with simulated mipmaps: {} ({}x{}, levels={})",
                       filename, tex.width, tex.height, tex.mip_levels);
    textures_.push_back(std::move(tex));
    mip_info_buffer_.reset();
    return static_cast<int>(textures_.size() - 1);
}


int TextureManager::LoadHDRTexture(const std::string& filename, float intensity = 1.0) {
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
            tex.hdr_data[i*4 + 0] = img[i*3 + 0] * intensity;
            tex.hdr_data[i*4 + 1] = img[i*3 + 1] * intensity;
            tex.hdr_data[i*4 + 2] = img[i*3 + 2] * intensity;
            tex.hdr_data[i*4 + 3] = 1.0f; 
        }
    } else if (tex.channels == 4) {
        memcpy(tex.hdr_data.data(), img, pixel_count * 4 * sizeof(float));
        for (size_t i = 0; i < pixel_count; ++i) {
            tex.hdr_data[i*4 + 0] *= intensity;
            tex.hdr_data[i*4 + 1] *= intensity;
            tex.hdr_data[i*4 + 2] *= intensity;
        }
    }
    
    stbi_image_free(img);
    
    auto mip_chain = BuildMipChainRGBA32(tex.hdr_data.data(), tex.width, tex.height);
    tex.mip_levels = (int)mip_chain.size();
    tex.has_mipmap = (tex.mip_levels > 1);
    int w = tex.width, h = tex.height;
    for (int level = 0; level < tex.mip_levels; ++level)
    {
        std::shared_ptr<grassland::graphics::Image> image;
        core_->CreateImage(w, h, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &image);
        image->UploadData(mip_chain[level].data());
        tex.mipImages.push_back(image);
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
    }
    tex.gpuImage = tex.mipImages[0];
    
    grassland::LogInfo("HDR Texture loaded: {} ({}x{}, {} channels)", 
                      filename, tex.width, tex.height, tex.channels);

    ComputeHDRDistribution(tex);

    textures_.push_back(std::move(tex));
    mip_info_buffer_.reset();
    return static_cast<int>(textures_.size() - 1);
}

float TextureManager::ComputeLuminance(const glm::vec3& rgb) const {
    return glm::dot(rgb, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

void TextureManager::ComputeHDRDistribution(Texture& texture) {
    if (texture.width <= 0 || texture.height <= 0 || texture.hdr_data.empty()) {
        return;
    }
    
    grassland::LogInfo("Computing HDR distribution for {}x{} texture", 
                      texture.width, texture.height);
    
    HDRDistributionData& distribution = texture.hdr_distribution;
    int width = texture.width;
    int height = texture.height;
    
    distribution.luminance_map.resize(width * height);
    std::vector<float> row_sums(height, 0.0f);
    
    const float PI = 3.14159265359f;
    float total_luminance = 0.0f;

    for (int y = 0; y < height; ++y) {
        float row_sum = 0.0f;
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            glm::vec3 color(
                texture.hdr_data[idx],
                texture.hdr_data[idx + 1],
                texture.hdr_data[idx + 2]
            );
            
            float theta = (static_cast<float>(y) + 0.5f) * PI / height;
            float sin_theta = std::sin(theta);
            
            float luminance = ComputeLuminance(color) * sin_theta;
            distribution.luminance_map[y * width + x] = luminance;
            row_sum += luminance;
        }
        row_sums[y] = row_sum;
        total_luminance += row_sum;
    }
    
    distribution.total_luminance = total_luminance * (PI / height) * (2.0f * PI / width); // with area
    
    // 2. 计算条件CDF（每行）
    distribution.conditional_cdf.resize(width * height);
    for (int y = 0; y < height; ++y) {
        float cumulative = 0.0f;
        if (row_sums[y] > 0.0f) {
            for (int x = 0; x < width; ++x) {
                cumulative += distribution.luminance_map[y * width + x] / row_sums[y];
                distribution.conditional_cdf[y * width + x] = cumulative;
            }
        } else {
            // 如果行总和为0，使用均匀分布
            for (int x = 0; x < width; ++x) {
                distribution.conditional_cdf[y * width + x] = (x + 1.0f) / width;
            }
        }
    }
    
    // 3. 计算边缘CDF（行方向）
    distribution.marginal_cdf.resize(height);
    float cumulative = 0.0f;
    if (total_luminance > 0.0f) {
        for (int y = 0; y < height; ++y) {
            cumulative += row_sums[y] / total_luminance;
            distribution.marginal_cdf[y] = cumulative;
        }
    } else {
        // 如果总亮度为0，使用均匀分布
        for (int y = 0; y < height; ++y) {
            distribution.marginal_cdf[y] = (y + 1.0f) / height;
        }
    }
    
    // 验证CDF
    if (!distribution.marginal_cdf.empty()) {
        float max_cdf = distribution.marginal_cdf.back();
        if (std::abs(max_cdf - 1.0f) > 1e-6f) {
            grassland::LogWarning("HDR marginal CDF normalization error: {}", max_cdf);
            // 重新归一化
            for (float& val : distribution.marginal_cdf) {
                val /= max_cdf;
            }
        }
    }

    size_t buffer_size = sizeof(float) * (3 + width * height + height); // [width, height, total_luminance] + conditional_cdf + marginal_cdf
    std::vector<float> cdf_data;
    cdf_data.reserve(3 + width * height + height);
    
    // 存储宽度和高度
    cdf_data.push_back(static_cast<float>(width));
    cdf_data.push_back(static_cast<float>(height));
    cdf_data.push_back(distribution.total_luminance);
    
    // 存储条件CDF
    cdf_data.insert(cdf_data.end(), 
                   distribution.conditional_cdf.begin(), 
                   distribution.conditional_cdf.end());
    
    // 存储边缘CDF
    cdf_data.insert(cdf_data.end(),
                   distribution.marginal_cdf.begin(),
                   distribution.marginal_cdf.end());
    
    // 创建GPU缓冲区
    core_->CreateBuffer(buffer_size, 
                       grassland::graphics::BUFFER_TYPE_DYNAMIC,
                       &distribution.cdf_buffer);
    
    distribution.cdf_buffer->UploadData(cdf_data.data(), buffer_size);
    
    texture.has_hdr_distribution = true;
    
    grassland::LogInfo("HDR distribution computed: total luminance = {:.3f}", 
                      total_luminance);
}

void TextureManager::PreprocessHDRDistribution(int texture_id) {
    if (texture_id < 0 || texture_id >= static_cast<int>(textures_.size())) {
        throw std::runtime_error("Invalid texture ID for HDR preprocessing");
    }
    
    Texture& texture = textures_[texture_id];
    if (texture.hdr_data.empty()) {
        throw std::runtime_error("Texture is not HDR");
    }
    
    if (!texture.has_hdr_distribution) {
        ComputeHDRDistribution(texture);
    }
}

float TextureManager::GetHDRTotalLuminance(int texture_id) const {
    if (texture_id < 0 || texture_id >= static_cast<int>(textures_.size())) {
        return 0.0f;
    }
    
    const Texture& texture = textures_[texture_id];
    if (!texture.has_hdr_distribution) {
        return 0.0f;
    }
    
    return texture.hdr_distribution.total_luminance;
}

Texture* TextureManager::GetTexture(int id) {
    if (id < 0 || id >= (int)textures_.size()) return nullptr;
    return &textures_[id];
}


void TextureManager::CollectMipImages(std::vector<grassland::graphics::Image*>& outImages) const {
    outImages.clear();
    for (const auto& tex : textures_) {
        if (tex.mipImages.empty()) continue;
        for (const auto& img : tex.mipImages) {
            outImages.push_back(img.get());
        }
    }
}

void TextureManager::BuildMipInfo(std::vector<MipInfo>& outInfos) const {
    outInfos.clear();
    uint32_t runningStart = 0;
    for (const auto& tex : textures_) {
        MipInfo info{};
        info.start = runningStart;
        info.levels = (uint32_t)tex.mip_levels;
        outInfos.push_back(info);
        runningStart += info.levels;
    }
}

grassland::graphics::Buffer* TextureManager::GetOrCreateMipInfoBuffer() {
    if (mip_info_buffer_) return mip_info_buffer_.get();

    std::vector<MipInfo> infos;
    BuildMipInfo(infos);

    if (infos.empty()) {
        // 保证至少有一个元素，避免空缓冲绑定
        infos.push_back({0u, 0u});
    }

    size_t sz = infos.size() * sizeof(MipInfo);
    core_->CreateBuffer(sz, grassland::graphics::BUFFER_TYPE_DYNAMIC, &mip_info_buffer_);
    mip_info_buffer_->UploadData(infos.data(), sz);

    return mip_info_buffer_.get();
}
