#pragma once
#include "long_march.h"
#include "Scene.h" 
#define MUL_INTENS 8
#define MUL_INTENS_SKYBOX 0.25

struct Light {
    glm::vec3 position;       // pos
    glm::vec3 direction;      // direction vector:
                             // - point light: not used
                             // - area light: normal direction
                             // - spot light: light direction
    glm::vec3 tangent;        // tangent vector of area light
    glm::vec3 color;          // light color
    float intensity;          // intensity
    glm::vec2 size;           // area light size)
    float radius;             // sphere light radius
    float cone_angle;         // spot light cone angle (degrees, cannot be 0)
    int type;                 // 0=point light, 1=area light, 2=spot light, 3=sphere light
    int enabled;             
    int visible;             

    Light(): position(0.0f), direction(0.0f,-1.0f,0.0f), 
            tangent(1.0f,0.0f,0.0f), 
            color(1.0f), intensity(1.0f), size(1.0f,1.0f), 
            radius(0.5f), type(0), enabled(true), visible(false), cone_angle(0.0f){}

    static Light CreatePointLight(const glm::vec3& pos, const glm::vec3& col=glm::vec3(1.0f),
                                  float intens=1.0f, bool visible=false)
    {
        Light light;
        light.position = pos;
        light.direction = glm::vec3(0.0f,0.0f,0.0f);
        light.tangent = glm::vec3(0.0f,0.0f,0.0f);
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.type = 0;
        light.enabled = true;
        light.visible = visible;
        return light;
    }
    
    static Light CreateAreaLight(const glm::vec3& pos, const glm::vec3& norm, 
                                 const glm::vec3& tangent, const glm::vec2& sz,
                                 const glm::vec3& col=glm::vec3(1.0f), float intens=1.0f,
                                 bool visible=false)
    {
        Light light;
        light.position = pos;
        light.direction = glm::normalize(norm);
        light.tangent = glm::normalize(tangent);
        
        if (std::abs(glm::dot(light.direction, light.tangent)) > 0.001f) {
            light.tangent = glm::normalize(light.tangent - glm::dot(light.tangent, light.direction) * light.direction);
        }
        
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.size = sz;
        light.type = 1;
        light.enabled = true;
        light.visible = visible;
        return light;
    }
    
    static Light CreateSpotLight(const glm::vec3& pos, const glm::vec3& dir,
                                        const glm::vec3& col=glm::vec3(1.0f), float intens=1.0f,
                                        float angle=45.0f, bool visible=false)
    {
        Light light;
        light.position = pos;
        light.direction = glm::normalize(dir);
        light.tangent = glm::vec3(0.0f,0.0f,0.0f);
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.type = 2; 
        light.enabled = true;
        light.visible = visible;
        light.cone_angle = angle;
        return light;
    }

    static Light CreateSphereLight(const glm::vec3& pos, float radius,
                                   const glm::vec3& col=glm::vec3(1.0f), float intens=1.0f,
                                   bool visible=false)
    {
        Light light;
        light.position = pos;
        light.radius = radius;
        light.direction = glm::vec3(0.0f,0.0f,0.0f);
        light.tangent = glm::vec3(0.0f,0.0f,0.0f);
        light.color = col;
        light.intensity = intens * MUL_INTENS;
        light.type = 3; // 球光源类型
        light.enabled = true;
        light.visible = visible;
        return light;
    }
};

class LightManager {
public:
    LightManager();
    ~LightManager();
    
    void Initialize(grassland::graphics::Core* core, Scene* scene = nullptr);
    
    // Create entity if visible=1
    void AddLight(const Light& light);
    
    void RemoveLight(size_t index);
    
    size_t GetLightCount() const { return lights_.size(); }
    
    const std::vector<Light>& GetLights() const { return lights_; }
    std::vector<Light>& GetLights() { return lights_; }
    
    void UpdateLight(size_t index, const Light& light);
    
    grassland::graphics::Buffer* GetLightsBuffer() const { return lights_buffer_.get(); }
    
    void UpdateBuffers();
    
    void CreateDefaultLights();
    
    int GetEnabledLightCount() const;

    void SetScene(Scene* scene) { scene_ = scene; }
    
    const std::vector<std::shared_ptr<Entity>>& GetLightEntities() const { return light_entities_; }

    static float CalculateLightPower(const Light& light);
    
    float GetTotalPower() const { return total_power_; }
    
    const std::vector<float>& GetPowerWeights() const { return power_weights_; }
    grassland::graphics::Buffer* GetPowerWeightsBuffer() const { return power_weights_buffer_.get(); }

    void SetHDRPower(float hdr_power){ hdr_power_ = hdr_power; }

private:
    std::vector<Light> lights_;
    std::vector<std::shared_ptr<Entity>> light_entities_;
    std::unique_ptr<grassland::graphics::Buffer> lights_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> light_count_buffer_;
    grassland::graphics::Core* core_;
    Scene* scene_;
    bool buffers_initialized_;

    std::vector<float> power_weights_;
    float total_power_;                         
    std::unique_ptr<grassland::graphics::Buffer> power_weights_buffer_; 

    void UpdatePowerData();

    std::shared_ptr<Entity> CreateLightEntity(const Light& light, size_t light_index);

    glm::mat4 CalculateAreaLightTransform(const Light& light) const;

    float hdr_power_;
};