#pragma once
#include "long_march.h"
#include "Material.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

class Texture {
public:
    Texture(grassland::graphics::Core* core);
    ~Texture();
    // 文件纹理
    bool LoadFromFile(const std::string& file_path);
    // 内存数据
    bool LoadFromMemory(const void* data,uint32_t width,uint32_t height, 
                       grassland::graphics::ImageFormat format);
    // 纯色纹理
    bool CreateSolidColor(const glm::vec4& color,uint32_t width=1,uint32_t height=1);
    // 获取GPU图像对象
    grassland::graphics::Image* GetImage() const{return image_.get();}
    // 获取纹理信息
    uint32_t GetWidth() const{return width_;}
    uint32_t GetHeight() const{return height_;}
    grassland::graphics::ImageFormat GetFormat() const{return format_;}
    // 检查纹理是否有效
    bool IsValid() const{return image_!=nullptr;}
private:
    grassland::graphics::Core* core_;
    std::unique_ptr<grassland::graphics::Image> image_;
    uint32_t width_;
    uint32_t height_;
    grassland::graphics::ImageFormat format_;
};

// 纹理管理器类
class TextureManager {
public:
    TextureManager(grassland::graphics::Core* core);
    ~TextureManager();
    // 加载纹理并返回ID
    int LoadTexture(const std::string& file_path);
    // 从内存创建纹理
    int CreateTextureFromMemory(const std::string& name,const void* data, 
                               uint32_t width,uint32_t height,
                               grassland::graphics::ImageFormat format);
    // 创建纯色纹理
    int CreateSolidColorTexture(const std::string& name,const glm::vec4& color, 
                               uint32_t width = 1,uint32_t height=1);
    // 获取纹理
    Texture* GetTexture(int texture_id);
    Texture* GetTexture(const std::string& name);
    // 获取纹理ID
    int GetTextureID(const std::string& name) const;
    // 获取所有纹理
    const std::vector<std::unique_ptr<Texture>>& GetAllTextures() const{return textures_;}
    // 获取默认纹理（白色）
    int GetDefaultWhiteTexture() const{return default_white_texture_id_;} 
    // 获取默认法线纹理（(0.5, 0.5, 1.0)）
    int GetDefaultNormalTexture() const{return default_normal_texture_id_;}
    // 获取默认黑色纹理
    int GetDefaultBlackTexture() const{return default_black_texture_id_;}
    // 获取默认粗糙度纹理（0.5）
    int GetDefaultRoughnessTexture() const{return default_roughness_texture_id_;}
    // 获取默认金属度纹理（0.0）
    int GetDefaultMetallicTexture() const{return default_metallic_texture_id_;}
    // 创建纹理描述符表（用于绑定到着色器）
    bool CreateDescriptorTable();
    // 获取描述符表
    grassland::graphics::DescriptorTable* GetDescriptorTable() const{return descriptor_table_.get();}
    // 清理所有纹理
    void Clear();
    
private:
    grassland::graphics::Core* core_;
    std::vector<std::unique_ptr<Texture>> textures_;
    std::unordered_map<std::string, int> texture_name_to_id_;
    std::unique_ptr<grassland::graphics::DescriptorTable> descriptor_table_;
    // 默认纹理ID
    int default_white_texture_id_;
    int default_normal_texture_id_;
    int default_black_texture_id_;
    int default_roughness_texture_id_;
    int default_metallic_texture_id_;
    
    // 创建默认纹理
    void CreateDefaultTextures();
};