#include "Scene.h"

Scene::Scene(grassland::graphics::Core* core)
    : core_(core) {
}

Scene::~Scene() {
    Clear();
}

void Scene::AddEntity(std::shared_ptr<Entity> entity) {
    if (!entity || !entity->IsValid()) {
        grassland::LogError("Cannot add invalid entity to scene");
        return;
    }

    // Build BLAS for the entity
    entity->BuildBLAS(core_);
    
    entities_.push_back(entity);
    grassland::LogInfo("Added entity to scene (total: {})", entities_.size());
}

void Scene::Clear() {
    entities_.clear();
    tlas_.reset();
    materials_buffer_.reset();
}

void Scene::BuildAccelerationStructures() {
    if (entities_.empty()) {
        grassland::LogWarning("No entities to build acceleration structures");
        return;
    }

    // Create TLAS instances from all entities
    std::vector<grassland::graphics::RayTracingInstance> instances;
    instances.reserve(entities_.size());

    for (size_t i = 0; i < entities_.size(); ++i) {
        auto& entity = entities_[i];
        if (entity->GetBLAS()) {
            // Create instance with entity's transform
            // instanceCustomIndex is used to index into materials buffer
            // Convert mat4 to mat4x3 (drop the last row which is always [0,0,0,1] for affine transforms)
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());
            
            auto instance = entity->GetBLAS()->MakeInstance(
                transform_3x4,
                static_cast<uint32_t>(i),  // instanceCustomIndex for material lookup
                0xFF,                       // instanceMask
                0,                          // instanceShaderBindingTableRecordOffset
                grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE
            );
            instances.push_back(instance);
        }
    }

    // Build TLAS
    core_->CreateTopLevelAccelerationStructure(instances, &tlas_);
    grassland::LogInfo("Built TLAS with {} instances", instances.size());

    // Update materials buffer
    UpdateMaterialsBuffer();

    // *add
    BuildGeometryBuffers();
}

void Scene::UpdateInstances() {
    if (!tlas_ || entities_.empty()) {
        return;
    }

    // Recreate instances with updated transforms
    std::vector<grassland::graphics::RayTracingInstance> instances;
    instances.reserve(entities_.size());

    for (size_t i = 0; i < entities_.size(); ++i) {
        auto& entity = entities_[i];
        if (entity->GetBLAS()) {
            // Convert mat4 to mat4x3
            glm::mat4x3 transform_3x4 = glm::mat4x3(entity->GetTransform());
            
            auto instance = entity->GetBLAS()->MakeInstance(
                transform_3x4,
                static_cast<uint32_t>(i),
                0xFF,
                0,
                grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE
            );
            instances.push_back(instance);
        }
    }

    // Update TLAS
    tlas_->UpdateInstances(instances);
}

void Scene::UpdateMaterialsBuffer() {
    if (entities_.empty()) {
        return;
    }

    // Collect all materials
    std::vector<Material> materials;
    materials.reserve(entities_.size());

    for (const auto& entity : entities_) {
        materials.push_back(entity->GetMaterial());
    }

    // Create/update materials buffer
    size_t buffer_size = materials.size() * sizeof(Material);
    
    if (!materials_buffer_) {
        core_->CreateBuffer(buffer_size, 
                          grassland::graphics::BUFFER_TYPE_DYNAMIC, 
                          &materials_buffer_);
    }
    
    materials_buffer_->UploadData(materials.data(), buffer_size);
    grassland::LogInfo("Updated materials buffer with {} materials", materials.size());
}

// *add
// 几何信息 Buffer，静态

struct GeometryDescriptor {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t vertex_count;
    uint32_t index_count;
    uint32_t material_index;
};

void Scene::BuildGeometryBuffers() {
    if (entities_.empty()) {
        return;
    }

    size_t total_vertices = 0;
    size_t total_indices = 0;

    for (const auto& entity : entities_) {
        total_vertices += entity->GetVertexCount();
        total_indices += entity->GetIndexCount();
    }

    // 2. 创建几何描述符数组
    std::vector<GeometryDescriptor> geometry_descriptors;
    geometry_descriptors.reserve(entities_.size());

    // 3. 创建合并的顶点/法线/索引数据
    std::vector<glm::vec3> all_vertices;
    std::vector<uint32_t> all_indices;
    
    all_vertices.reserve(total_vertices);
    all_indices.reserve(total_indices);

    // 4. 合并所有实体的几何数据
    uint32_t vertex_offset = 0;
    uint32_t index_offset = 0;

    for (size_t i = 0; i < entities_.size(); ++i) {
        const auto& entity = entities_[i];
        
        // 下载实体几何数据
        size_t vertex_count = entity->GetVertexCount();
        size_t index_count = entity->GetIndexCount();

        std::vector<glm::vec3> vertices(vertex_count);
        std::vector<glm::vec3> normals(vertex_count);
        std::vector<uint32_t> indices(index_count);

        entity->GetVertexBuffer()->DownloadData(vertices.data());
        entity->GetIndexBuffer()->DownloadData(indices.data());

        // 添加到全局缓冲区
        all_vertices.insert(all_vertices.end(), vertices.begin(), vertices.end());
        
        // 调整索引（加上当前偏移）
        for (auto index : indices) {
            all_indices.push_back(index + vertex_offset);
        }

        // 创建几何描述符
        GeometryDescriptor desc{};
        desc.vertex_offset = vertex_offset;
        desc.index_offset = index_offset;
        desc.vertex_count = static_cast<uint32_t>(vertex_count);
        desc.index_count = static_cast<uint32_t>(index_count);
        desc.material_index = static_cast<uint32_t>(i);
        
        geometry_descriptors.push_back(desc);

        vertex_offset += vertex_count;
        index_offset += index_count;
    }

    // 5. 创建GPU缓冲区
    size_t vertex_buffer_size = all_vertices.size() * sizeof(glm::vec3);
    size_t index_buffer_size = all_indices.size() * sizeof(uint32_t);
    size_t descriptors_size = geometry_descriptors.size() * sizeof(GeometryDescriptor);

    core_->CreateBuffer(vertex_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &global_vertex_buffer_);
    core_->CreateBuffer(index_buffer_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &global_index_buffer_);
    core_->CreateBuffer(descriptors_size, grassland::graphics::BUFFER_TYPE_DYNAMIC, &geometry_descriptors_buffer_);

    // 6. 上传数据
    global_vertex_buffer_->UploadData(all_vertices.data(), vertex_buffer_size);
    global_index_buffer_->UploadData(all_indices.data(), index_buffer_size);
    geometry_descriptors_buffer_->UploadData(geometry_descriptors.data(), descriptors_size);

    grassland::LogInfo("Built geometry buffers: {} vertices, {} indices across {} entities",
                      all_vertices.size(),  all_indices.size(), entities_.size());
}

