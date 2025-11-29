
struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

struct Material {
  float3 base_color;
  float roughness;
  float metallic;

  float3 emission;        // 自发光颜色
  float ior;                // 折射率
  float transparency;        
  int material_type;        // 材质类型: 0=漫反射, 1=镜面, 2=玻璃, 3=发射
  int texture_id;       
};

struct HoverInfo {
  int hovered_entity_id;
};

RaytracingAccelerationStructure as : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);
StructuredBuffer<Material> materials : register(t0, space3);
ConstantBuffer<HoverInfo> hover_info : register(b0, space4);
RWTexture2D<int> entity_id_output : register(u0, space5);
RWTexture2D<float4> accumulated_color : register(u0, space6);
RWTexture2D<int> accumulated_samples : register(u0, space7);

struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
};

#define MAX_DEPTH 8
#define RR_THRESHOLD 0.9f
#define t_min 0.001
#define t_max 10000.0
#define eps 5e-4

void wanghash(inout uint seed)
{
  seed=(seed^61)^(seed>>16);
  seed=seed*9;
  seed=seed^(seed>>4);
  seed=seed*0x27d4eb2d;
  seed=seed^(seed>>15);
}

float random(inout uint seed)
{
  wanghash(seed);
  return float(seed)/4294967296.0;
}

float f3_max(float3 u){
  return max(u[0], max(u[1], u[2]));
}

void SampleBSDF(Material material, float3 ray, float3 normal, out float3 wi, out float pdf, inout uint seed){
  // do something
}

[shader("raygeneration")] void RayGenMain() {
  
  float2 pixel_center = (float2)DispatchRaysIndex() + float2(0.5,0.5); //float2(random(), random());
  float2 uv = pixel_center / float2(DispatchRaysDimensions().xy);
  uv.y = 1.0 - uv.y;
  float2 d = uv * 2.0 - 1.0;
  float4 origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
  float4 target = mul(camera_info.screen_to_camera, float4(d, 1, 1));
  float4 direction = mul(camera_info.camera_to_world, float4(target.xyz, 0));

  uint2 pixel_coords = DispatchRaysIndex().xy;

  int seed = pixel_coords[0] * pixel_coords[1];

  int depth = 0;
  float3 color = float3(0.0, 0.0, 0.0);
  float3 throughout = float3(1.0, 1.0, 1.0);

  RayDesc ray;
  ray.Origin = origin.xyz;
  ray.Direction = normalize(direction.xyz);
  ray.TMin = t_min;
  ray.TMax = t_max;

  entity_id_output[pixel_coords] =  -1;

  while(1){
    ++depth;

    RayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.hit = false;
    payload.instance_id = 0;

    TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    if(!payload.hit){
      color = payload.color * throughout;
      break;
    }

    if(depth == 1){
      // Write entity ID to the ID buffer
      // If no hit, write -1; otherwise write the instance ID
      entity_id_output[pixel_coords] = (int)payload.instance_id;
    }

    if(depth > 4){
      float p_survive = min(f3_max(throughout), RR_THRESHOLD);
      if(random(seed) > p_survive){
        //
        break;
      }
      throughout /= p_survive;

    }

    Material material = materials[payload.instance_id];

    color += throughout * material.emission;

    float pdf, wi;
    float3 bsdf = SampleBSDF(material, -ray.Direction, payload.normal, wi, pdf, seed);

    if(pdf <= 0.0) break;

    float cosTheta = dot(wi, payload.normal);

    throughout *= bsdf * abs(cosTheta) / pdf;

    ray.Origin = payload.hit_point + (cosTheta > 0 ? eps : -eps) * payload.normal;
    ray.Direction = wi;
    ray.TMin = t_min;
    ray.TMax = t_max;

  }
  
  // Write to immediate output (for camera movement mode)
  output[pixel_coords] = float4(color, 1);
  
  // Accumulate color for progressive rendering (when camera is stationary)
  float4 prev_color = accumulated_color[pixel_coords];
  int prev_samples = accumulated_samples[pixel_coords];
  
  accumulated_color[pixel_coords] = prev_color + float4(color, 1);
  accumulated_samples[pixel_coords] = prev_samples + 1;

}

[shader("miss")] void MissMain(inout RayPayload payload) {
  // Sky gradient
  float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
  payload.color = lerp(float3(1.0, 1.0, 1.0), float3(0.5, 0.7, 1.0), t);
  payload.hit = false;
  payload.instance_id = 0xFFFFFFFF; // Invalid ID for miss
}

[shader("closesthit")] void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  payload.hit = true;
  
  // Get material index from instance
  uint material_idx = InstanceID();
  payload.instance_id = material_idx;
  
  // Load material
  Material mat = materials[material_idx];
  
  // Simple diffuse lighting
  float3 world_normal = normalize(float3(0, 1, 0)); // Placeholder, should compute from geometry
  float3 light_dir = normalize(float3(1, 1, 1));
  float ndotl = max(0.0, dot(world_normal, light_dir));
  
  // Apply material color (NO hover highlighting here - done in post-process)
  float3 diffuse = mat.base_color * (0.3 + 0.7 * ndotl);
  
  payload.color = diffuse;
}