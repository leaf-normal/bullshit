// Texture.cpp
#include "Texture.h"
#include "grassland/util/image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
Texture::Texture(grassland::graphics::Core* core):
    core_(core),width_(0),height_(0),
    format_(grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM){
}
Texture::~Texture(){
}
bool Texture::LoadFromFile(const std::string& file_path)
{
    // 使用 grassland 的图像加载功能
    std::string full_path=grassland::FindAssetFile(file_path);
    int width,height,channels;
    stbi_uc* data=stbi_load(full_path.c_str(),&width,&height,&channels,4);
    if(!data)
    {
        grassland::LogError("Failed to load texture: {}",file_path);
        return false;
    }
    bool result=LoadFromMemory(data,width,height,
                               grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM);
    stbi_image_free(data);
    if(result)
    {
        grassland::LogInfo("Loaded texture: {} ({}x{})",file_path,width,height);
    }
    return result;
}
bool Texture::LoadFromMemory(const void* data,uint32_t width,uint32_t height,
                            grassland::graphics::ImageFormat format) {
    width_=width;
    height_=height;
    format_=format;
    // 创建GPU图像
    core_->CreateImage(width,height,format,&image_);
    // 上传数据
    size_t pixel_size=0;
    switch(format)
    {
        case grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM:
            pixel_size=4;
            break;
        case grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT:
            pixel_size=16;
            break;
        case grassland::graphics::IMAGE_FORMAT_R8_UNORM:
            pixel_size=1;
            break;
        default:
            pixel_size=4; // 默认为4字节
            break;
    }
    size_t data_size=width*height*pixel_size;
    image_->UploadData(data,data_size);
    return true;
}
bool Texture::CreateSolidColor(const glm::vec4& color,uint32_t width,uint32_t height) {
    width_=width;
    height_=height;
    format_=grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM;
    // 创建颜色数据
    std::vector<uint8_t> color_data(width*height*4);
    for (size_t i=0;i<width*height;i++) {
        color_data[i*4+0]=static_cast<uint8_t>(color.r*255.0f);
        color_data[i*4+1]=static_cast<uint8_t>(color.g*255.0f);
        color_data[i*4+2]=static_cast<uint8_t>(color.b*255.0f);
        color_data[i*4+3]=static_cast<uint8_t>(color.a*255.0f);
    }
    // 创建GPU图像
    core_->CreateImage(width,height,format_,&image_);
    image_->UploadData(color_data.data(),color_data.size());
    return true;
}

TextureManager::TextureManager(grassland::graphics::Core* core):
    core_(core),default_white_texture_id_(-1),default_normal_texture_id_(-1)
    default_black_texture_id_(-1),default_roughness_texture_id_(-1),default_metallic_texture_id_(-1){
    // 创建默认纹理
    CreateDefaultTextures();
}
TextureManager::~TextureManager() {
    Clear();
}
void TextureManager::CreateDefaultTextures() {
    // 创建白色纹理 (1x1, RGBA = (1,1,1,1))
    default_white_texture_id_=CreateSolidColorTexture("_default_white",glm::vec4(1.0f,1.0f,1.0f,1.0f));
    // 创建黑色纹理 (1x1, RGBA = (0,0,0,1))
    default_black_texture_id_=CreateSolidColorTexture("_default_black",glm::vec4(0.0f,0.0f,0.0f,1.0f));
    // 创建法线纹理 (1x1, RGBA = (0.5,0.5,1,1))
    default_normal_texture_id_=CreateSolidColorTexture("_default_normal",glm::vec4(0.5f,0.5f,1.0f,1.0f));
    // 创建粗糙度纹理 (1x1, RGBA = (0.5,0.5,0.5,1))
    default_roughness_texture_id_=CreateSolidColorTexture("_default_roughness",glm::vec4(0.5f,0.5f,0.5f,1.0f));
    // 创建金属度纹理 (1x1, RGBA = (0,0,0,1))
    default_metallic_texture_id_=CreateSolidColorTexture("_default_metallic",glm::vec4(0.0f,0.0f,0.0f,1.0f));
}
int TextureManager::LoadTexture(const std::string& file_path)
{
    // 检查是否已加载
    auto it=texture_name_to_id_.find(file_path);
    if(it!=texture_name_to_id_.end())
    {
        return it->second;
    }
    // 创建新纹理
    auto texture=std::make_unique<Texture>(core_);
    if(!texture->LoadFromFile(file_path))
    {
        grassland::LogError("Failed to load texture: {}",file_path);
        return -1;
    }
    int texture_id=static_cast<int>(textures_.size());
    texture_name_to_id_[file_path]=texture_id;
    textures_.push_back(std::move(texture));
    return texture_id;
}
int TextureManager::CreateTextureFromMemory(const std::string& name,const void* data,uint32_t width,
                                           uint32_t height,grassland::graphics::ImageFormat format) {
    // 检查是否已存在
    auto it=texture_name_to_id_.find(name);
    if(it!=texture_name_to_id_.end())
    {
        return it->second;
    }
    auto texture=std::make_unique<Texture>(core_);
    if(!texture->LoadFromMemory(data, width, height, format))
    {
        grassland::LogError("Failed to create texture from memory: {}",name);
        return -1;
    }
    int texture_id=static_cast<int>(textures_.size());
    texture_name_to_id_[name]=texture_id;
    textures_.push_back(std::move(texture));
    return texture_id;
}
int TextureManager::CreateSolidColorTexture(const std::string& name,const glm::vec4& color, 
                                           uint32_t width,uint32_t height)
{
    auto it=texture_name_to_id_.find(name);
    if(it!=texture_name_to_id_.end())
    {
        return it->second;
    }
    auto texture=std::make_unique<Texture>(core_);
    if(!texture->CreateSolidColor(color,width,height))
    {
        grassland::LogError("Failed to create solid color texture: {}",name);
        return -1;
    }
    int texture_id=static_cast<int>(textures_.size());
    texture_name_to_id_[name]=texture_id;
    textures_.push_back(std::move(texture));
    return texture_id;
}
Texture* TextureManager::GetTexture(int texture_id)
{
    if(texture_id<0||texture_id>=static_cast<int>(textures_.size()))
    {
        return nullptr;
    }
    return textures_[texture_id].get();
}
Texture* TextureManager::GetTexture(const std::string& name) {
    auto it=texture_name_to_id_.find(name);
    if(it==texture_name_to_id_.end())
    {
        return nullptr;
    }
    return GetTexture(it->second);
}
int TextureManager::GetTextureID(const std::string& name) const {
    auto it=texture_name_to_id_.find(name);
    if (it==texture_name_to_id_.end())
    {
        return -1;
    }
    return it->second;
}
bool TextureManager::CreateDescriptorTable() {
    if(!core_)
    {
        return false;
    }
    // 收集所有纹理图像
    std::vector<grassland::graphics::Image*> texture_images;
    texture_images.reserve(textures_.size());
    for(const auto& texture:textures_)
    {
        if(texture&&texture->GetImage())
        {
            texture_images.push_back(texture->GetImage());
        }else{
            // 如果纹理无效，使用默认白色纹理
            texture_images.push_back(textures_[default_white_texture_id_]->GetImage());
        }
    }
    core_->CreateDescriptorTable(texture_images, &descriptor_table_);
    grassland::LogInfo("Created texture descriptor table with {} textures", texture_images.size());
    return true;
}

void TextureManager::Clear()
{
    textures_.clear();
    texture_name_to_id_.clear();
    descriptor_table_.reset();
}