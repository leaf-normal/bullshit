#pragma once
#include "long_march.h"
#include "Entity.h"
#include "Material.h"
#include "Geometry.h"
#include "Motion.h"
#include <vector>
#include <memory>

#define MAX_MOTION_GROUPS 4

// Scene manages a collection of entities and builds the TLAS

struct MotionGroup {
    MotionParams motion;                    // 该组的运动参数
    std::vector<std::shared_ptr<Entity>> entities;  // 属于该组的实体
    std::vector<int> entity_id_;  // 运动组ID到实体ID的映射
    std::vector<grassland::graphics::RayTracingInstance> instances;  // 实例数据
};

class Scene {
public:
    Scene(grassland::graphics::Core* core);
    ~Scene();

    void AddEntity(std::shared_ptr<Entity> entity);

    void Clear();

    void BuildAccelerationStructures();

    grassland::graphics::AccelerationStructure* GetTLAS(int group_id = 0) const {
        if (group_id < 0 || group_id >= tlas_array_.size()) {
            return empty_tlas_.get();
        }
        return tlas_array_[group_id] ? tlas_array_[group_id].get() : empty_tlas_.get();
    }

    const std::vector<MotionGroup>& GetMotionGroups() const { return motion_groups_; }
    
    size_t GetMotionGroupCount() const { return motion_groups_.size(); }

    grassland::graphics::Buffer* GetMotionGroupsBuffer() const { return motion_groups_buffer_.get(); }

    grassland::graphics::Buffer* GetMaterialsBuffer() const { return materials_buffer_.get(); }

    const std::vector<std::shared_ptr<Entity>>& GetEntities() const { return entities_; }

    size_t GetEntityCount() const { return entities_.size(); }

    grassland::graphics::Buffer* GetVertexInfoBuffer() const { return global_vertex_info_buffer_.get(); }
    grassland::graphics::Buffer* GetIndexBuffer() const { return global_index_buffer_.get(); }
    grassland::graphics::Buffer* GetGeometryDescriptorsBuffer() const { return geometry_descriptors_buffer_.get(); }

private:
    void UpdateMaterialsBuffer();
    void BuildGeometryBuffers();
    void BuildMotionGroupsBuffer(); 
    
    int CalculateMotionGroup(const MotionParams& motion);
    
    void BuildGroupTLAS(int group_id);
    
    grassland::graphics::Core* core_;
    std::vector<std::shared_ptr<Entity>> entities_;
    
    std::vector<MotionGroup> motion_groups_;

    std::unique_ptr<grassland::graphics::Buffer> motion_groups_buffer_;
    std::vector<std::unique_ptr<grassland::graphics::AccelerationStructure>> tlas_array_;
    std::unique_ptr<grassland::graphics::AccelerationStructure> empty_tlas_;

    std::unique_ptr<grassland::graphics::Buffer> materials_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> geometry_descriptors_buffer_;
    std::unique_ptr<grassland::graphics::Buffer> global_vertex_info_buffer_; 
    std::unique_ptr<grassland::graphics::Buffer> global_index_buffer_;
};