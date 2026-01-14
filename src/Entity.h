#pragma once
#include "long_march.h"
#include "Material.h"
#include "Geometry.h"
#include "Motion.h"

// Entity represents a mesh instance with a material and transform
class Entity {
public:
    Entity(const std::string& obj_file_path, 
           const Material& material = Material(),
           const glm::mat4& transform = glm::mat4(1.0f),
           const MotionParams& motion = MotionParams());

    ~Entity();

    // Load mesh from OBJ file
    bool LoadMesh(const std::string& obj_file_path);

    // Getters
    grassland::graphics::Buffer* GetVertexInfoBuffer() const { return vertex_info_buffer_.get(); }
    grassland::graphics::Buffer* GetIndexBuffer() const { return index_buffer_.get(); }

    const Material& GetMaterial() const { return material_; }
    const glm::mat4& GetTransform() const { return transform_; }
    grassland::graphics::AccelerationStructure* GetBLAS() const { return blas_.get(); }

    size_t GetVertexCount() const { return mesh_loaded_ ? mesh_.NumVertices() : 0; } 
    size_t GetIndexCount() const { return mesh_loaded_ ? mesh_.NumIndices() : 0; } 


    // Setters
    void SetMaterial(const Material& material) { material_ = material; }
    void SetTransform(const glm::mat4& transform) { transform_ = transform; }

    // Create BLAS
    void BuildBLAS(grassland::graphics::Core* core);

    bool IsValid() const { return mesh_loaded_; }

    // Motion
    const MotionParams& GetMotionParams() const { return motion_params_; }
    void SetMotionParams(const MotionParams& motion) { motion_params_ = motion; }
    void SetMotionGroups(const int idx){
        motion_params_.group_id = idx;
        material_.group_id;
    }
    
    // Transform at time t
    glm::mat4 GetTransformAtTime(float t) const {
        return motion_params_.GetTransformAtTime(t, transform_);
    }    

private:
    grassland::Mesh<float> mesh_;
    Material material_;
    glm::mat4 transform_;

    std::unique_ptr<grassland::graphics::Buffer> vertex_info_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> index_buffer_;

    std::unique_ptr<grassland::graphics::AccelerationStructure> blas_;

    bool mesh_loaded_;

    MotionParams motion_params_;
};

