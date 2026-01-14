#pragma once
#include "long_march.h"
#include <string>
#include <vector>
#include <memory>
// #include "grassland/graphics/Core.h"
// #include "grassland/graphics/Image.h"

struct HDRDistributionData {
    std::vector<float> marginal_cdf;      
    std::vector<float> conditional_cdf;   
    float total_luminance;
    std::vector<float> luminance_map;

    std::shared_ptr<grassland::graphics::Buffer> cdf_buffer;
};

struct Texture {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;      // LDR
    std::vector<float> hdr_data;          // HDR
    std::shared_ptr<grassland::graphics::Image> gpuImage;

    std::vector<std::shared_ptr<grassland::graphics::Image>> mipImages;
    int mip_levels;
    bool has_mipmap=false;

    HDRDistributionData hdr_distribution;
    bool has_hdr_distribution;           
};

struct MipInfo{
    uint32_t start;
    uint32_t levels;
};

class TextureManager {
public:
    explicit TextureManager(grassland::graphics::Core* core) : core_(core) {}
    int LoadTexture(const std::string& filename);
    int LoadHDRTexture(const std::string& filename, float intensity);  
    Texture* GetTexture(int id);
    int TextureCount() const { return static_cast<int>(textures_.size()); }


    void PreprocessHDRDistribution(int texture_id); 
    float GetHDRTotalLuminance(int texture_id) const;
    grassland::graphics::Buffer* GetHDRCDFBuffer(int texture_id) {
        if (texture_id < 0 || texture_id >= static_cast<int>(textures_.size())) {
            return nullptr;
        }
        
        Texture& texture = textures_[texture_id];
        if (!texture.has_hdr_distribution) {
            return nullptr;
        }
        
        return texture.hdr_distribution.cdf_buffer.get();
    }
    
    void CollectMipImages(std::vector<grassland::graphics::Image*>& outImages) const;
    void BuildMipInfo(std::vector<MipInfo>& outInfos) const;
    grassland::graphics::Buffer* GetOrCreateMipInfoBuffer();


private:
    grassland::graphics::Core* core_;
    std::vector<Texture> textures_;

    void ComputeHDRDistribution(Texture& texture);
    float ComputeLuminance(const glm::vec3& rgb) const;

    static std::vector<std::vector<unsigned char>> BuildMipChainRGBA8(const unsigned char* base, int width, int height);
    static std::vector<std::vector<float>> BuildMipChainRGBA32(const float* base, int width, int height);
    std::shared_ptr<grassland::graphics::Buffer> mip_info_buffer_;
};
