// TextureManager.h 修改
#pragma once
#include "long_march.h"
#include <string>
#include <vector>
#include <memory>
// #include "grassland/graphics/Core.h"
// #include "grassland/graphics/Image.h"

struct HDRDistributionData {
    std::vector<float> marginal_cdf;      // 行方向的边缘CDF (高度)
    std::vector<float> conditional_cdf;   // 每行的条件CDF (宽度 × 高度)
    float total_luminance;               // 总亮度（功率）
    std::vector<float> luminance_map;     // 亮度图（用于调试和采样）

    std::shared_ptr<grassland::graphics::Buffer> cdf_buffer;
};

struct Texture {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;      // 用于LDR纹理
    std::vector<float> hdr_data;          // 用于HDR纹理
    std::shared_ptr<grassland::graphics::Image> gpuImage;

    HDRDistributionData hdr_distribution; // HDR分布数据
    bool has_hdr_distribution;            // 是否已计算分布数据
};

class TextureManager {
public:
    explicit TextureManager(grassland::graphics::Core* core) : core_(core) {}
    int LoadTexture(const std::string& filename);
    int LoadHDRTexture(const std::string& filename, float intensity);  
    Texture* GetTexture(int id);
    int TextureCount() const { return static_cast<int>(textures_.size()); }


    void PreprocessHDRDistribution(int texture_id);   // 预处理HDR分布
    float GetHDRTotalLuminance(int texture_id) const; // 获取HDR总亮度
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

private:
    grassland::graphics::Core* core_;
    std::vector<Texture> textures_;

    // HDR分布计算辅助函数
    void ComputeHDRDistribution(Texture& texture);
    float ComputeLuminance(const glm::vec3& rgb) const;
};
