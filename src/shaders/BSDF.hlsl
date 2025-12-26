struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
  float focal_distance;     // 焦点距离（世界空间）
  float aperture_size;      // 光圈直径（控制模糊强度）
  float focal_length;       // 焦距（控制视角）
  float lens_radius;        // 透镜半径 = aperture_size/2
  int enable_depth_of_field;

  float3 camera_linear_velocity;  // 相机线性速度
  float camera_angular_velocity; // 相机角速度（绕相机方向的旋转）
  int enable_motion_blur;            // 是否启用运动模糊
  float exposure_time;    
};

struct Material {
  float3 base_color;
  float roughness;
  float metallic;
  uint light_index;
  
  float3 emission;          // 自发光颜色
  float ior;                // 折射率
  float transparency;       // 透明度 
  int texture_id;
  int normal_tex_id;
  int attribute_tex_id;
  
  float subsurface;     //次表面散射
  float specular;      //镜面反射强度
  float specular_tint; //镜面反射
  float anisotropic;   //各向异性[-1,1]
  float sheen;         //光泽层
  float sheen_tint;    //光泽层染色
  float clearcoat;     //清漆层强度
  float clearcoat_roughness; //清漆层粗糙度

  uint group_id;
};

struct HoverInfo {
  int hovered_entity_id;
};

// ====================== 常量缓冲区定义 ======================

#define MAX_MOTION_GROUPS 1

RaytracingAccelerationStructure as : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);
StructuredBuffer<Material> materials : register(t0, space3);
ConstantBuffer<HoverInfo> hover_info : register(b0, space4);
RWTexture2D<int> entity_id_output : register(u0, space5);
RWTexture2D<float4> accumulated_color : register(u0, space6);
RWTexture2D<int> accumulated_samples : register(u0, space7);

struct GeometryDescriptor {
    uint vertex_offset;
    uint index_offset;
    uint vertex_count;
    uint index_count;
};

struct VertexInfo {
    float3 position;
    float3 normal;    
    float2 texcoord;
};

StructuredBuffer<GeometryDescriptor> geometry_descriptors : register(t0, space8);
StructuredBuffer<VertexInfo> vertices : register(t0, space9);
StructuredBuffer<uint> indices : register(t0, space10);

struct RenderSettings {
    uint frame_count;
    uint samples_per_pixel;
    uint max_depth;
    uint enable_accumulation;
    uint light_count;
    int skybox_texture_id_;            // If not able, set to -1

    float2 resolution;                 // width * height
};

ConstantBuffer<RenderSettings> render_setting : register(b0, space11);

// ====================== 光源系统 ======================
#define MAX_LIGHTS 32

struct Light {
    float3 position;       // 光源位置
    float3 direction;      // 方向向量：
                             // - 点光源：不使用（置0）
                             // - 面光源：法线方向（单向）
                             // - 聚光灯：光照方向
    float3 tangent;        // 面光源的切向量，保证正交归一化
    float3 color;          // 光源颜色
    float intensity;          // 辐射亮度（面光源，球光源） / 辐射强度（点光源，聚光灯）
    float2 size;           // 面光源尺寸(矩形-长宽)
    float radius;             // 球光源半径
    float cone_angle; 
    uint type;                 // 0=点光源, 1=面光源, 2=聚光灯, 3=球光源
    uint enabled;             // 光源是否启用
    uint visible;             // 光源是否可见
};

StructuredBuffer<Light> lights : register(t0, space12);                // hdr is not included in lights
StructuredBuffer<float> light_power_weights : register(t0, space13);   // Range from [0, light_count + 1), with power weight of hdr added to last position

// 物体运动

struct MotionParams {
    float3 linear_velocity;
    float3 angular_velocity;
    float3 pivot_point;          
    int is_static;
    int group_id; 
};
StructuredBuffer<MotionParams> motion_groups : register(t0, space14);

// Texture
Texture2D<float4> g_Textures[128] : register(t0, space18);
SamplerState g_Sampler : register(s0, space19);

// hdr_cdf
StructuredBuffer<float> hdr_cdf : register(t0, space20);   // [width, height, total_luminance] + conditional_cdf + marginal_cdf


struct RayDifferential {
    float3 rxOrigin;    // origin for ray at pixel + (1,0)
    float3 ryOrigin;    // origin for ray at pixel + (0,1)
    float3 rxDirection; // direction for ray at pixel + (1,0)
    float3 ryDirection; // direction for ray at pixel + (0,1)
    bool hasDifferentials;
};
struct RayPayload {
  float3 color;
  bool hit;
  uint instance_id;
  float3 normal;
  float3 hit_point;
  bool is_filp;
  float4 attribute;

  RayDifferential diffs;

  // 新增：纹理导数（u,v 对 x,y 的偏导）
  float dudx;
  float dudy;
  float dvdx;
  float dvdy;

  // 新增：位置与法线微分（可选，供高级 shading 使用）
  float3 dpdx;
  float3 dpdy;
  float3 dndx;
  float3 dndy;

  // 协方差 tracer 新增字段
  float3x3 dirCov;  // 当前方向的协方差 Cov(L)
  bool hasDirCov;   // 是否有效

};

// ====================== 常量定义 ======================
#define MAX_DEPTH 1
#define RR_THRESHOLD 0.95f
#define t_min 1e-3
#define t_max 10000.0
#define eps 5e-4 // used for geometry
#define EPS 1e-6 // used for division
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define INV_PI 0.31830988618

// ====================== 随机数生成 ======================
uint pcg_hash(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(inout uint seed) {
    return (float)pcg_hash(seed) / 4294967296.0;
}

float2 random2(inout uint seed) {
    return float2(random(seed), random(seed));
}

uint generate_seed(uint2 pixel, uint frame) {
    return (pixel.x * 1973u + pixel.y * 9277u + frame * 26699u) ^ 0x6f5ca34du;
}

// ====================== 光源采样系统 ======================
struct LightSample {
    float3 position;
    float3 direction;
    float3 radiance;
    float pdf;
    int light_index;
    bool valid;
};

bool is_enable_lights(inout uint light_count){

  if(render_setting.skybox_texture_id_ >= 0) return true;
  
  for(int i = 0; i < light_count; ++i){
    if(light_power_weights[i] > 0.0){
      return true;
    }
  }
  return false;
}

// 按功率重要性采样光源
int sample_light_by_power(inout uint light_count, inout uint seed) {
    
    // 使用预计算的权重进行采样
    float u = random(seed);
    uint last = 0;
    
    for (int i = 0; i < light_count; i++) {
        if(light_power_weights[i] > 0.0){
          u -= light_power_weights[i];
          if (u <= 0) {
              return i;
          }
          last = i;    
        }

    }
    
    return render_setting.skybox_texture_id_ >= 0 ? light_count : last;
}

void sample_point_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled) return;
    
    float3 to_light = light.position - hit_point;
    float distance = length(to_light);
    
    if (distance < 1e-6) return;
    
    sample.position = light.position;
    sample.direction = to_light / distance;
    
    // 物理正确的辐射亮度（朗伯体点光源）
    sample.radiance = light.color * light.intensity / (distance * distance + 1e-6);
    
    sample.pdf = 1;
    sample.light_index = -1; // 将由调用者设置
    sample.valid = true;
}

// 面光源采样 - 使用切向量
void sample_area_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled || light.size.x <= 0.0 || light.size.y <= 0.0) return;
    
    // 构建局部坐标系
    float3 bitangent = normalize(cross(light.direction, light.tangent));

    // 在矩形上均匀采样（面积采样）
    float3 local_pos = float3((random(seed) - 0.5) * light.size.x, 
                              (random(seed) - 0.5) * light.size.y, 
                              0.0);
    
    // 转换到世界空间
    sample.position = light.position + 
                     local_pos.x * light.tangent + 
                     local_pos.y * bitangent;
    
    // 计算方向、距离、余弦项
    float3 to_light = sample.position - hit_point;
    float distance = length(to_light);
    sample.direction = to_light / distance;
    
    // 检查可见性（面光源只有一面发光）
    float cos_theta_l = dot(-sample.direction, light.direction);
    if (cos_theta_l <= 0.0) {
        sample.valid = 0;
        return; // 从背面看不到
    }
    
    // 几何项
    float area = light.size.x * light.size.y;
    
    // 物理正确的辐射亮度（朗伯体面光源）
    sample.radiance = light.color * light.intensity;
    
    // PDF转换：面积PDF -> 立体角PDF
    float solid_angle_pdf = (distance * distance) / (cos_theta_l * area);
    sample.pdf = solid_angle_pdf;
    
    sample.light_index = -1;
    sample.valid = (sample.pdf > 0.0);
}

// 球光源采样 - 新增
void sample_sphere_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled || light.radius <= 0.0) return;
    
    // 在球面上均匀采样
    float z = 2.0 * random(seed) - 1.0;
    float phi = TWO_PI * random(seed);
    float r = sqrt(max(0.0, 1.0 - z * z));
    
    float3 sphere_dir = float3(r * cos(phi), r * sin(phi), z);
    
    // 转换到世界空间
    sample.position = light.position + light.radius * sphere_dir;
    
    // 计算方向、距离、余弦项
    float3 to_light = sample.position - hit_point;
    float distance = length(to_light);
    sample.direction = to_light / distance;
    
    // 球面法线（指向外）
    float3 sphere_normal = sphere_dir;
    float cos_theta_l = max(0.0, dot(-sample.direction, sphere_normal));
    if (cos_theta_l <= 0.0) {
        return; // 从背面看不到
    }
    
    // 几何项
    float surface_area = 4.0 * PI * light.radius * light.radius;
    
    // 物理正确的辐射亮度（朗伯体球光源）
    sample.radiance = light.color * light.intensity;
    
    // PDF转换：面积PDF -> 立体角PDF
    float area_pdf = 1.0 / surface_area;
    float solid_angle_pdf = (distance * distance) / (cos_theta_l * surface_area);
    sample.pdf = solid_angle_pdf;
    
    sample.light_index = -1;
    sample.valid = (sample.pdf > 0.0);
}

void sample_spot_light(inout Light light, inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (!light.enabled) return;
    
    float3 to_light = light.position - hit_point;
    float distance = length(to_light);
    
    if (distance < 1e-6) return;

    sample.position = light.position;
    sample.direction = normalize(to_light);
    
    // 锥角检查
    float cos_angle = dot(-sample.direction, light.direction);
    float cos_cutoff = cos(radians(light.cone_angle * 0.5));

    if (cos_angle < cos_cutoff) {
        return; // 在锥角外
    }
    
    // 平滑过渡
    float epsilon = 0.1 * (1.0 - cos_cutoff);
    float spot_factor = smoothstep(cos_cutoff, cos_cutoff + epsilon, cos_angle);
    
    // // 物理正确的辐射亮度
    float solid_angle = 2.0 * PI * (1.0 - cos_cutoff);
    sample.radiance = light.color * light.intensity * spot_factor / (distance * distance + 1e-6);
    
    sample.pdf = 1.0;  
    sample.light_index = -1;
    sample.valid = true;
}

uint binary_search_cdf(float rand_value, uint start_idx, uint count) {
    uint left = 0;
    uint right = count - 1;
    uint result = 0;
    
    while (left <= right) {
        uint mid = (left + right) >> 1;  // 除以2的位运算优化
        float cdf_value = hdr_cdf[start_idx + mid];
        
        if (rand_value <= cdf_value) {
            result = mid;
            if (mid == 0 || rand_value > hdr_cdf[start_idx + mid - 1]) {
                break;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

void sample_hdr(inout float3 hit_point, inout uint seed, out LightSample sample) {
    sample = (LightSample)0;
    
    if (render_setting.skybox_texture_id_ < 0) {
        return;
    }
    
    // 从CDF缓冲区读取宽度和高度
    uint width = (uint)hdr_cdf[0];
    uint height = (uint)hdr_cdf[1];
    float hdr_total_luminance = hdr_cdf[2];
    
    uint cdf_offset = 3; 
    
    // 1. 使用二分查找采样边缘CDF（行）
    uint marginal_cdf_start = cdf_offset + (uint)(width * height);
    uint row = binary_search_cdf(random(seed), marginal_cdf_start, height);
    
    // 2. 使用二分查找采样条件CDF（列）
    uint conditional_cdf_start = cdf_offset + row * (uint)width;
    uint col = binary_search_cdf(random(seed), conditional_cdf_start, width);
    
    float u = (float(col) + 0.5) / width;
    float v = (float(row) + 0.5) / height;
    
    float phi = u * TWO_PI;
    float theta = v * PI;
    
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    
    float3 direction = float3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
    
    float3 hdr_color = g_Textures[render_setting.skybox_texture_id_].SampleLevel(g_Sampler, float2(u, v), 0).rgb;
    float luminance = dot(hdr_color, float3(0.2126, 0.7152, 0.0722));
    
    float solid_angle_pdf = luminance / hdr_total_luminance;
    
    // 5. 设置采样结果
    sample.position = hit_point + direction * 10000.0; // 远点
    sample.direction = direction;
    sample.radiance = hdr_color;
    sample.pdf = solid_angle_pdf;
    sample.valid = (solid_angle_pdf > 0.0) && (luminance > 0.0);
}

// ====================== MIS核心 ======================

// 平衡启发式 MIS 权重
float mis_balance_weight(float pdf_a, float pdf_b) {
    float total = pdf_a + pdf_b ;
    return total > 0.0 ? pdf_a / total : 0.0;
}

// 功率启发式 MIS 权重
float mis_power_weight(float pdf_a, float pdf_b) {
    float w = pdf_a * pdf_a ; // β=2
    float total = w + pdf_b * pdf_b ;
    return total > 0.0 ? w / total : 0.0;
}

// ====================== BSDF辅助函数 ======================

float f3_max(inout float3 u){
  return max(u[0], max(u[1], u[2]));
}
float sqr(float x)
{
  return x*x;
}
float3 sqr(float3 x)
{
  return x*x;
}
bool Refract(float3 v, float3 n, float eta, out float3 t) {
    float c = dot(v, n);
    float k = 1.0 - eta * eta * (1.0 - c * c);
    if (k < 0.0) return false; // 全反射 (TIR)
    t = eta * v - (eta * c + sqrt(k)) * n;
    return true;
}

float SchlickWeight(float cosTheta)
{
  return pow( clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
float SchlickFresnelScalar(float F0, float cosTheta)
{
  return F0 + (1.0 - F0) * SchlickWeight(cosTheta);
}
float3 SchlickFresnel(float3 F0, float cosTheta)
{
  return F0 + (1.0 - F0) * SchlickWeight(cosTheta);
}
float GTR1(float NdotH, float a)
{
  if(a >= 1.0) return INV_PI;
  float a2 = sqr(a);
  float t = 1.0 + (a2 - 1.0) * sqr(NdotH);
  return (a2 - 1.0) / (PI * log(a2 + EPS) * t + EPS);
}
float GTR2(float NdotH, float a)
{
    float a2 = sqr(a);
    float t = 1.0 + (a2 - 1.0) * sqr(NdotH);
    return a2 / (PI * sqr(t) + EPS);
}
float GTR2_Anisotropic(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1.0 / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH) + EPS);
}

float SmithG_GGX(float NdotV, float alphaG) {
    float a=sqr(alphaG);
    float b=sqr(NdotV);
    return 2.0*NdotV/(NdotV+sqrt(a+(1.0-a)*b) + EPS);
}
float SmithG_GGX_Anisotropic(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 2.0*NdotV/(NdotV+sqrt(sqr(VdotX*ax)+sqr(VdotY*ay)+sqr(NdotV)) + EPS);
}
float SmithG_GGX_Correlated(float NdotV, float NdotL, float alpha)
{
    NdotV = max(NdotV, EPS);
    NdotL = max(NdotL, EPS);
    float a2 = alpha * alpha;
    float lambdaV = (-1.0 + sqrt(1.0 + a2 * (1.0 - NdotV * NdotV) / (NdotV * NdotV))) * 0.5;
    float lambdaL = (-1.0 + sqrt(1.0 + a2 * (1.0 - NdotL * NdotL) / (NdotL * NdotL))) * 0.5;
    float G = 1.0 / (1.0 + lambdaV + lambdaL);
    return G;
}
float SmithG_GGX_Correlated_Anisotropic(float NdotV,float NdotL,float VdotX,float VdotY,float LdotX,float LdotY,float ax,float ay)
{
    NdotV = max(NdotV, EPS);
    NdotL = max(NdotL, EPS);
    float lambdaV = (-1.0 + sqrt(1.0 + (sqr(VdotX * ax) + sqr(VdotY * ay)) / sqr(NdotV))) * 0.5;
    float lambdaL = (-1.0 + sqrt(1.0 + (sqr(LdotX * ax) + sqr(LdotY * ay)) / sqr(NdotL))) * 0.5;
    float G = 1.0 / (1.0 + lambdaV + lambdaL);
    return G;
}
float3 SampleHemisphereCos(float2 randd, inout float3 normal)
{
  float r=sqrt(randd.x);
  float theta=2.0*PI*randd.y;
  float x=r*cos(theta);
  float y=r*sin(theta);
  float z=sqrt(max(0.0,1.0-randd.x));
  float3 up=abs(normal.z)<1-eps?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  float3 tangent=normalize(cross(up, normal));
  float3 bitangent=cross(normal,tangent);
  return normalize(x*tangent+y*bitangent+z*normal);
}

void GetTangent(inout float3 normal, out float3 tangent, out float3 bitangent)
{
  float3 up=abs(normal.z)<0.999?float3(0.0,0.0,1.0):float3(1.0,0.0,0.0);
  tangent=normalize(cross(up,normal));
  bitangent=cross(normal,tangent);
}

float3 SampleGGX_VNDF_Isotropic(float3 V_world, float3 N_world, float roughness,float2 u)
{
    float3 T, B;
    GetTangent(N_world, T, B);
    float3 V = normalize(float3(dot(V_world, T), dot(V_world, B), dot(V_world, N_world)));
    float alpha = max(EPS, roughness);
    float3 Vh = normalize(float3(alpha * V.x, alpha * V.y, V.z));
    float3 up = (abs(Vh.z) < 1.0 - EPS) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T1 = normalize(cross(up, Vh));
    float3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s  = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 H = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    float3 H_world = normalize(H.x * T + H.y * B + H.z * N_world);
    return H_world;
}

float3 SampleGGX_Anisotropic(float3 ray, float roughness, float anisotropic, float2 rd, float3 normal, float3 tangent)
{

    float3 bitangent = cross(normal, tangent);

    float3 V_local = normalize(float3(dot(ray, tangent), dot(ray, bitangent), dot(ray, normal)));

    float aspect = sqrt(1.0 - 0.9 * anisotropic);
    float ax = max(eps, roughness / aspect);
    float ay = max(eps, roughness * aspect);
    float3 Vh = normalize(float3(ax * V_local.x, ay * V_local.y, V_local.z));
    float3 up = abs(Vh.z) < 1.0 - eps ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T1 = normalize(cross(up, Vh));
    float3 T2 = cross(Vh, T1);
    float r = sqrt(rd.x);
    float phi = 2.0 * PI * rd.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    float Nh_len = sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2));
    float3 Nh = t1 * T1 + t2 * T2 + Nh_len * Vh;
    float3 H_local = normalize(float3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
    float3 H_world = H_local.x * tangent + H_local.y * bitangent + H_local.z * normal;
    return H_world;
}

float3 SampleGGX_Distribution(float roughness, float2 rd, inout float3 normal, inout float3 tangent, inout float3 bitangent)
{
    float a = sqr(roughness);
    float a2 = sqr(a);
    float phi = 2.0 * PI * rd.y;
    float cos_theta = sqrt((1.0 - rd.x) / (1.0 - rd.x + a2 * rd.x));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float3 H_local = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    float3 H_world = H_local.x * tangent + H_local.y * bitangent + H_local.z * normal;
    return normalize(H_world);
}


// ====================== Adaptive mipmap 辅助函数 =============== 

// Solve 2x2 linear system A * x = b, where A = [a00 a01; a10 a11].
// 返回是否成功（行列式不太小），并把解写入 out_x (float2)。
// 若退化（|det| < tol），用正则化（加 small to diagonal）。
bool Solve2x2(float a00, float a01, float a10, float a11, float2 b, out float2 out_x)
{
    float det = a00 * a11 - a01 * a10;
    if (abs(det) < EPS) {
        // 正则化：加小量到对角线
        a00 += EPS; a11 += EPS;
        det = a00 * a11 - a01 * a10;
        if (abs(det) < EPS) {
            out_x = float2(0.0, 0.0);
            return false;
        }
    }
    float invDet = 1.0 / det;
    out_x.x = ( a11 * b.x - a01 * b.y) * invDet;
    out_x.y = (-a10 * b.x + a00 * b.y) * invDet;
    return true;
}

// Project 3D vector b onto basis (e1,e2) solving b = e1 * x + e2 * y in least-squares sense.
// 返回 (x,y) 解（float2）。若基向量接近共线，会退化并返回稳定解。
float2 SolveForCoeffs3x2(float3 e1, float3 e2, float3 b)
{
    // 构造 2x2 Gram 矩阵 G = [e1·e1 e1·e2; e2·e1 e2·e2] 和 rhs = [e1·b; e2·b]
    float a00 = dot(e1, e1);
    float a01 = dot(e1, e2);
    float a10 = a01;
    float a11 = dot(e2, e2);
    float2 rhs = float2(dot(e1, b), dot(e2, b));
    float2 sol;
    if (!Solve2x2(a00, a01, a10, a11, rhs, sol)) {
        // 退化：选择最大能量方向投影
        if (a00 >= a11 && a00 > 0.0) {
            sol.x = rhs.x / (a00 + 1e-8);
            sol.y = 0.0;
        } else if (a11 > 0.0) {
            sol.x = 0.0;
            sol.y = rhs.y / (a11 + 1e-8);
        } else {
            sol = float2(0.0, 0.0);
        }
    }
    return sol;
}

// ====================== Cov 矩阵计算 ==============================

// J_L_n = ∂L/∂n_m for reflection L = d - 2 (n_m·d) n_m
// Inputs: d (incident direction, pointing to surface? use same convention as reflect), n_m (micro normal, normalized)
// Output: 3x3 matrix as float3x3
float3x3 ReflectionJacobian_wrt_Normal(float3 d, float3 n_m)
{
    // scalar
    float nd = dot(n_m, d);
    // outer product n_m * d^T
    float3x3 nm_dt = float3x3(
        n_m.x * d.x, n_m.x * d.y, n_m.x * d.z,
        n_m.y * d.x, n_m.y * d.y, n_m.y * d.z,
        n_m.z * d.x, n_m.z * d.y, n_m.z * d.z
    );
    // identity
    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);
    // J = -2 * ( n_m * d^T + nd * I )
    float3x3 J = -2.0 * (nm_dt + nd * I);
    return J;
}

// 反射方向对入射方向的雅可比 J_R_v = ∂R/∂v
// v: 入射方向（和 reflect(I,N) 的 I 一致）
// n: 法线（单位向量）
float3x3 ReflectionJacobian_wrt_Dir(float3 v, float3 n)
{
    // 外积 n n^T
    float3x3 nnT = float3x3(
        n.x * n.x, n.x * n.y, n.x * n.z,
        n.y * n.x, n.y * n.y, n.y * n.z,
        n.z * n.x, n.z * n.y, n.z * n.z
    );

    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);

    // J_R_v = I - 2 * n n^T
    float3x3 J = I - 2.0 * nnT;
    return J;
}

// J_T_n = ∂t/∂n_m for refraction t = eta * v - (eta*c + sqrt(k)) * n_m
// v: incident direction (与 Refract 中 v 一致的约定)
// n_m: micro normal (normalized)
// eta: IOR ratio (n1/n2 or n2/n1，与你 BSDF 中使用一致)
// 返回 3x3 雅可比矩阵；若处于 TIR 区间，返回零矩阵（由上层逻辑改为反射处理）
float3x3 RefractionJacobian_wrt_Normal(float3 v, float3 n_m, float eta)
{
    float c = dot(v, n_m);
    float eta2 = eta * eta;
    float k = 1.0 - eta2 * (1.0 - c * c);

    // 全反射或接近全反射：此时没有折射方向，这个分支通常会走反射 BSDF，
    // 在这里我们返回 0 矩阵，把传播交给反射路径。
    if (k <= 0.0) {
        return float3x3(0,0,0, 0,0,0, 0,0,0);
    }

    float s = sqrt(k); // = sqrt(1 - eta^2(1 - c^2))

    // dA/dn = (eta + eta^2 * c / s) * v
    float coeff = eta + (eta2 * c) / max(s, 1e-8);
    float3 dA_dn = coeff * v;

    // A = eta * c + s
    float A = eta * c + s;

    // outer product n_m * (dA_dn)^T
    float3x3 nm_dA = float3x3(
        n_m.x * dA_dn.x, n_m.x * dA_dn.y, n_m.x * dA_dn.z,
        n_m.y * dA_dn.x, n_m.y * dA_dn.y, n_m.y * dA_dn.z,
        n_m.z * dA_dn.x, n_m.z * dA_dn.y, n_m.z * dA_dn.z
    );

    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);

    // J_T_n = - ( n_m * (dA_dn)^T + A * I )
    float3x3 J = -(nm_dA + A * I);
    return J;
}

// 折射方向对入射方向的雅可比 J_T_v = ∂t/∂v
// v: 入射方向（和 Refract 中 v 的约定一致）
// n: 法线（单位向量）
// eta: 折射率比（和 BSDF 中一致，例如 1/ior 或 ior）
float3x3 RefractionJacobian_wrt_Dir(float3 v, float3 n, float eta)
{
    float c    = dot(v, n);
    float eta2 = eta * eta;
    float k    = 1.0 - eta2 * (1.0 - c * c);

    // 全反射或接近 TIR：此时不该走折射路径，返回单位矩阵，交由反射分支处理更合理
    if (k <= 0.0) {
        return float3x3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        );
    }

    float s = sqrt(k);

    // coeff = η + η² c / sqrt(k)
    float coeff = eta + (eta2 * c) / max(s, 1e-8);

    float3x3 nnT = float3x3(
        n.x * n.x, n.x * n.y, n.x * n.z,
        n.y * n.x, n.y * n.y, n.y * n.z,
        n.z * n.x, n.z * n.y, n.z * n.z
    );

    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);

    // J_T_v = η I - coeff * (n n^T)
    float3x3 J = eta * I - coeff * nnT;
    return J;
}


// Cov_n = sigma_n^2 * (I - n n^T)
// n: shading normal (world space), sigma_n: scalar (approx alpha)
float3x3 MicroNormalCovariance(float3 n, float sigma_n)
{
    float3x3 I = float3x3(1,0,0, 0,1,0, 0,0,1);
    // n n^T
    float3x3 nnT = float3x3(
        n.x*n.x, n.x*n.y, n.x*n.z,
        n.y*n.x, n.y*n.y, n.y*n.z,
        n.z*n.x, n.z*n.y, n.z*n.z
    );
    float3x3 cov = sigma_n * sigma_n * (I - nnT);
    return cov;
}

// 新：各向异性微法线协方差（world-space）
// T/B: triangle-level tangent / bitangent (world-space unit vectors)
// alpha_x / alpha_y: micro-normal std. dev. 沿 T/B（通常 alpha_x/alpha_y ~ alpha +/- anisotropy adjust）
float3x3 MicroNormalCovariance_Aniso(float3 n, float3 T, float3 B, float alpha_x, float alpha_y)
{
    // Cov_n = alpha_x^2 * (T * T^T) + alpha_y^2 * (B * B^T)
    float3x3 TT = float3x3(
        T.x*T.x, T.x*T.y, T.x*T.z,
        T.y*T.x, T.y*T.y, T.y*T.z,
        T.z*T.x, T.z*T.y, T.z*T.z
    );
    float3x3 BB = float3x3(
        B.x*B.x, B.x*B.y, B.x*B.z,
        B.y*B.x, B.y*B.y, B.y*B.z,
        B.z*B.x, B.z*B.y, B.z*B.z
    );
    float3x3 cov = (alpha_x*alpha_x) * TT + (alpha_y*alpha_y) * BB;
    // ensure no normal-axis variance (micro-normal fluctuation mainly in tangent plane)
    // small epsilon on diagonal to avoid exact singularity
    cov[0][0] += 1e-10; cov[1][1] += 1e-10; cov[2][2] += 1e-10;
    return cov;
}


// Cov_L = J * Cov_n * J^T
float3x3 PropagateNormalCovToDirectionCov(float3x3 J_L_n, float3x3 Cov_n)
{
    // temp = J * Cov_n
    float3x3 temp;
    // matrix multiply 3x3
    [unroll] for (int r = 0; r < 3; ++r) {
        [unroll] for (int c = 0; c < 3; ++c) {
            float s = 0.0;
            [unroll] for (int k = 0; k < 3; ++k) {
                s += J_L_n[r][k] * Cov_n[k][c];
            }
            temp[r][c] = s;
        }
    }
    // Cov_L = temp * J^T
    float3x3 CovL;
    [unroll] for (int r = 0; r < 3; ++r) {
        [unroll] for (int c = 0; c < 3; ++c) {
            float s = 0.0;
            [unroll] for (int k = 0; k < 3; ++k) {
                s += temp[r][k] * J_L_n[c][k]; // J^T: swap indices
            }
            CovL[r][c] = s;
        }
    }
    return CovL;
}

// Compute J_uv_L (2x3) for lat-long mapping and then Cov_uv = J * CovL * J^T
// Inputs: L (direction, normalized), CovL (3x3)
// Outputs: Cov_uv (2x2)
void DirectionCovToLatLongUVCov(float3 L, float3x3 CovL, out float2x2 CovUV)
{
    // components
    float lx = L.x;
    float ly = L.y;
    float lz = L.z;
    float denom_xz = lx*lx + lz*lz;
    denom_xz = max(denom_xz, 1e-12); // avoid division by zero at poles

    // ∂u/∂L = 1/(2π) * [ -lz/(lx^2+lz^2), 0, lx/(lx^2+lz^2) ]
    float inv2pi = 1.0 / (2.0 * PI);
    float3 du_dL = inv2pi * float3(-lz / denom_xz, 0.0, lx / denom_xz);

    // ∂v/∂L = -1/(π * sqrt(1 - ly^2)) * [0, 1, 0]
    float denom_v = sqrt(max(1.0 - ly*ly, 1e-12));
    float dv_dy = -1.0 / (PI * denom_v);
    float3 dv_dL = float3(0.0, dv_dy, 0.0);

    // Cov_uv = [du; dv] * CovL * [du; dv]^T
    // compute intermediate: a = CovL * du_dL, b = CovL * dv_dL
    float3 a = float3(
        CovL[0][0]*du_dL.x + CovL[0][1]*du_dL.y + CovL[0][2]*du_dL.z,
        CovL[1][0]*du_dL.x + CovL[1][1]*du_dL.y + CovL[1][2]*du_dL.z,
        CovL[2][0]*du_dL.x + CovL[2][1]*du_dL.y + CovL[2][2]*du_dL.z
    );
    float3 b = float3(
        CovL[0][0]*dv_dL.x + CovL[0][1]*dv_dL.y + CovL[0][2]*dv_dL.z,
        CovL[1][0]*dv_dL.x + CovL[1][1]*dv_dL.y + CovL[1][2]*dv_dL.z,
        CovL[2][0]*dv_dL.x + CovL[2][1]*dv_dL.y + CovL[2][2]*dv_dL.z
    );

    float c00 = dot(du_dL, a); // var(u)
    float c01 = dot(du_dL, b); // cov(u,v)
    float c11 = dot(dv_dL, b); // var(v)

    // pack into 2x2
    CovUV = float2x2(c00, c01, c01, c11);
}

// Convert CovUV (uv units) to texel units and compute LOD
// texture_id: environment texture id (lat-long texture width/height)
// Returns: lod (float), and optionally axis lengths & cov_texel (texel^2)
float ComputeEnvmapLOD_FromCovUV(int texture_id, float2x2 CovUV, out float2 axis_lengths, out float2x2 cov_texel)
{
    uint w, h;
    g_Textures[texture_id].GetDimensions(w, h);
    float2 texSize = float2(max(1u,w), max(1u,h));

    // Convert uv covariance to texel covariance: cov_texel = diag(texSize) * CovUV * diag(texSize)
    float2x2 D = float2x2(texSize.x, 0.0, 0.0, texSize.y);
    // cov_texel = D * CovUV * D
    float2x2 temp = float2x2(
        D[0][0]*CovUV[0][0] + D[0][1]*CovUV[1][0], D[0][0]*CovUV[0][1] + D[0][1]*CovUV[1][1],
        D[1][0]*CovUV[0][0] + D[1][1]*CovUV[1][0], D[1][0]*CovUV[0][1] + D[1][1]*CovUV[1][1]
    );
    cov_texel = float2x2(
        temp[0][0]*D[0][0] + temp[0][1]*D[1][0], temp[0][0]*D[0][1] + temp[0][1]*D[1][1],
        temp[1][0]*D[0][0] + temp[1][1]*D[1][0], temp[1][0]*D[0][1] + temp[1][1]*D[1][1]
    );

    // eigenvalues of cov_texel (2x2)
    float c00 = cov_texel[0][0];
    float c01 = cov_texel[0][1];
    float c11 = cov_texel[1][1];
    float trace = c00 + c11;
    float det = c00 * c11 - c01 * c01;
    float disc = max(0.0, trace * trace * 0.25 - det);
    float sqrt_disc = sqrt(disc);
    float lambda1 = trace * 0.5 + sqrt_disc;
    float lambda2 = trace * 0.5 - sqrt_disc;
    lambda1 = max(lambda1, 0.0);
    lambda2 = max(lambda2, 0.0);
    float r1 = sqrt(lambda1);
    float r2 = sqrt(lambda2);
    axis_lengths = float2(r1, r2);

    float rho = max(r1, r2);
    rho = max(rho, 1e-6);
    float lod = log2(rho);
    return lod;
}

// ========================== BSDF 主函数 =============================================

bool SampleBSDF(inout Material mat, inout float3 ray, inout float3 normal, out float3 wi, bool is_flip, inout uint seed){

  //lobe权重
  float diffuseweight=(1.0-mat.metallic)*(1.0-mat.transparency);//漫反射
  float specularweight=lerp(0.04, 1.0, mat.metallic)*(1.0-mat.transparency);
  float transmissionweight=mat.transparency*(1.0-mat.metallic);//透射
  float clearcoatweight=0.25*mat.clearcoat*(1.0-mat.transparency);//清漆层
  float sheenweight=mat.sheen*(1.0-mat.metallic)*(1.0-mat.transparency);//光泽层
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+sheenweight+clearcoatweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  sheenweight/=total;
  clearcoatweight/=total;
  float randLobe=random(seed);
  float3 tangent, bitangent;
  GetTangent(normal, tangent, bitangent);  
  if(randLobe<diffuseweight+sheenweight)//漫反射+简化光泽层模型
  {
    wi=SampleHemisphereCos(random2(seed),normal);
  }else if(randLobe<diffuseweight+specularweight+sheenweight)//镜面反射
  {
    float3 H;
    H=SampleGGX_Anisotropic(ray,mat.roughness,mat.anisotropic,random2(seed),normal,tangent); // replace
    wi=reflect(-ray,H);
    if(dot(normal,wi)<=0.0)return 0;
  }else if(randLobe<diffuseweight+specularweight+transmissionweight+sheenweight)//透射
  {
    float eta=(!is_flip)?1.0/mat.ior:mat.ior;
    float3 H=SampleGGX_Distribution(mat.roughness, random2(seed), normal, tangent, bitangent); //后续可以继续改为VNDF // replace
    if(Refract(-ray,H,eta,wi)==false)
    {
        wi=reflect(-ray,H);//全反射
        if(dot(normal,wi)<=0.0)return 0;
    }
  }else{//清漆
    float alpha_c=max(EPS, mat.clearcoat_roughness);
    float3 H=SampleGGX_Distribution(alpha_c,random2(seed),normal,tangent,bitangent);
    wi=reflect(-ray,H);
    if(dot(normal,wi)<=0.0)return 0;
  }
  wi=normalize(wi);
  return 1;
}

float3 EvalBSDF(inout Material mat, inout float3 ray, inout float3 wi, inout float3 normal, bool is_flip, out float pdf)
{
  float3 ret=float3(0.0,0.0,0.0);
  float3 tangent,bitangent;
  float3 F0=lerp(0.08*mat.specular,mat.base_color,mat.metallic);
  GetTangent(normal,tangent,bitangent);
  float Ndotray=dot(normal,ray);
  float Ndotwi=dot(normal,wi);
  bool is_trans=(Ndotray*Ndotwi)<0.0;
  if(Ndotray<=0.0)
  {
    pdf=0.0;
    return float3(1e9,0.0,0.0);
  }
  if(!is_trans&&Ndotwi<=0.0)
  {
    pdf=0.0;
    return float3(0.0,0.0,0.0);
  }
  float alpha=sqr(mat.roughness);
  float transmissionPDF=0.0;
  float specularPDF=0.0;
  float diffusePDF=0.0;
  float sheenPDF=0.0;
  float clearcoatPDF=0.0;
  
  //lobe权重
  float diffuseweight=(1.0-mat.metallic)*(1.0-mat.transparency);//漫反射
  float specularweight=lerp(0.04, 1.0, mat.metallic)*(1.0-mat.transparency);
  float transmissionweight=mat.transparency*(1.0-mat.metallic);//透射
  float sheenweight=mat.sheen*(1.0-mat.metallic)*(1.0-mat.transparency);//光泽层
  float clearcoatweight=0.25*mat.clearcoat*(1.0-mat.transparency);//清漆层  
  //normalization
  float total=diffuseweight+specularweight+transmissionweight+sheenweight+clearcoatweight;
  diffuseweight/=total;
  specularweight/=total;
  transmissionweight/=total;
  sheenweight/=total;
  clearcoatweight/=total;
  if(is_trans)//透射
  {
    float n1 = is_flip ? mat.ior : 1.0; // V 侧 (Camera 侧)
    float n2 = is_flip ? 1.0 : mat.ior; // L 侧 (Light 侧)
    float3 V = ray;
    float3 L = wi;
    float3 H = -(n1 * V + n2 * L);
    if (dot(H, H) < EPS) {
        pdf = 0.0;
        return float3(0.0, 0.0, 0.0);
    }
    H = normalize(H);
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);
    float NdotH = dot(normal, H);
    float NdotV = dot(normal, V);
    float NdotL = dot(normal, L);
    float D = GTR2(NdotH, mat.roughness); // 使用你现有的 GGX D
    float G = SmithG_GGX_Correlated(abs(NdotV), abs(NdotL), mat.roughness); // 你的 G
    float3 F_color = SchlickFresnel(F0, abs(VdotH)); 
    float3 T_color = 1.0 - F_color; // 透射颜色
    float sqrtDenom = n1 * VdotH + n2 * LdotH;
    float commonDenom = sqrtDenom * sqrtDenom + EPS;
    float factor = (abs(VdotH) * abs(LdotH) * n2 * n2) / (abs(NdotV) * abs(NdotL) * commonDenom);
    ret += mat.base_color * T_color * D * G * factor * transmissionweight;
    float jacobian = (n2 * n2 * abs(LdotH)) / commonDenom;
    transmissionPDF = D * abs(NdotH) * jacobian;

  }else{
    float3 H=normalize(ray+wi);
    float NdotH=dot(normal,H);
    float Hdotray=dot(H,ray);
    float3 F=SchlickFresnel(F0,Hdotray);
    float VdotH = dot(ray, H);
    float LdotH = dot(wi, H);
    float NdotV = dot(normal, ray);
    float NdotL = dot(normal, wi);
    //漫反射
    if(mat.metallic<1.0&&mat.transparency<1.0)
    {
      float FL=SchlickWeight(Ndotray);
      float FV=SchlickWeight(Ndotwi);
      float Fd90=0.5+2.0*mat.roughness*sqr(Hdotray);
      float Fd=lerp(1.0,Fd90,FL)*lerp(1.0,Fd90,FV);
      float3 diffuse=mat.base_color*(1.0-mat.metallic)*Fd/PI;
      ret+=diffuseweight*diffuse*(1.0-F);
    }
    diffusePDF=max(Ndotwi,0.0)/PI;

    // 镜面反射
    float ax=max(EPS,alpha/sqrt(1.0-0.9*mat.anisotropic));
    float ay=max(EPS,alpha*sqrt(1.0-0.9*mat.anisotropic));
    float D=GTR2_Anisotropic(NdotH,dot(H,tangent),dot(H,bitangent),ax,ay);
    float Vis=SmithG_GGX_Correlated_Anisotropic(Ndotray,Ndotwi,dot(ray,tangent), dot(ray, bitangent), dot(wi, tangent), dot(wi, bitangent), ax, ay);
    float3 spec=D*Vis*F; 
    if (mat.specular_tint > EPS)
    {
        float3 tint=lerp(float3(1.0, 1.0, 1.0), mat.base_color, mat.specular_tint);
        spec*=tint;
    }
    ret+=spec*specularweight;
    float aspect_eff = sqrt(1.0 - 0.9 * mat.anisotropic);
    float G1_V=SmithG_GGX_Anisotropic(Ndotray, dot(ray, tangent), dot(ray, bitangent), ax, ay);
    specularPDF=(D*G1_V)/(4.0*Ndotray+EPS);

    // 全反射
    if(transmissionweight > EPS)
    {
        float eta = (!is_flip) ? 1.0/mat.ior : mat.ior;
        float c = dot(ray, H);
        float k = 1.0 - eta * eta * (1.0 - c * c);
        if (k < 0.0) // 发生了全反射
        {
            float D_iso = GTR2(NdotH, mat.roughness);
            float pdf_h_tir = D_iso * abs(NdotH);
            ret+=spec*transmissionweight;
            transmissionPDF = pdf_h_tir / (4.0 * abs(Hdotray) + EPS);
        }
    }

    //光泽层
    if (mat.sheen > EPS && mat.metallic < 1.0 && mat.transparency < 1.0)
    {
        float3 base=mat.base_color;
        float lum = max(EPS, 0.2126 * base.x + 0.7152 * base.y + 0.0722 * base.z);
        float3 tintColor = base / lum; // 提取色相（去亮度）
        float3 sheenColor = lerp(float3(1.0,1.0,1.0), tintColor, mat.sheen_tint);
        float w = SchlickWeight(abs(Hdotray)); // 角度权重，常用 H·V
        float3 sheenTerm = sheenColor * (mat.sheen) * w;
        ret += sheenweight * sheenTerm * (1.0 - F);
        sheenPDF = diffusePDF; // 复用 cosine 采样的 PDF
    }
    //清漆层
    if (mat.clearcoat > EPS && mat.transparency < 1.0)
    {
        float alpha_c = max(EPS, mat.clearcoat_roughness);
        float Dc = GTR1(NdotH, alpha_c);
        float Gc = SmithG_GGX(NdotV, alpha_c) * SmithG_GGX(NdotL, alpha_c);
        float Fc = SchlickFresnelScalar(0.04, abs(Hdotray));
        float coat = (Dc * Gc * Fc) / (4.0 * abs(NdotV) * abs(NdotL) + EPS);
        ret = (1.0 - mat.clearcoat * Fc) * ret + clearcoatweight * coat;
        clearcoatPDF = (Dc * abs(NdotH)) / (4.0 * abs(Hdotray) + EPS);
    }
  }
  pdf=diffusePDF*diffuseweight+specularPDF*specularweight+transmissionPDF*transmissionweight+sheenPDF*sheenweight+clearcoatPDF*clearcoatweight;
  return ret;
}

// ====================== 阴影测试 ======================
bool test_shadow(inout float3 hit_point, inout float3 normal, inout float3 light_dir, inout float max_distance) {
    RayDesc shadow_ray;
    shadow_ray.Origin = hit_point;
    shadow_ray.Direction = light_dir;
    shadow_ray.TMin = t_min;
    shadow_ray.TMax = max_distance - eps * 3.0; // prevent hit the light
    
    RayPayload shadow_payload;
    
    // Traceray(shadow_ray, shadow_payload, true);
    shadow_payload.hit = true;
    TraceRay(as, 
          RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | 
          RAY_FLAG_FORCE_OPAQUE | 
          RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
          0xFF, 0, 1, 0, shadow_ray, shadow_payload);
    
    return shadow_payload.hit;
}

// ====================== MIS直接光照计算 ======================
float3 mis_direct_lighting(inout uint light_count, inout float3 hit_point, inout float3 normal, inout Material mat, inout float3 wo, bool is_flip, inout uint seed) {
    float3 total_light = float3(0, 0, 0);

    if (!is_enable_lights(light_count) || mat.metallic > 0.9) return total_light;

    uint sample_times = min(4, light_count + 1);

    for (int i = 0; i < sample_times; ++i) {
        // 按功率重要性采样光源
        int light_idx = sample_light_by_power(light_count, seed);
        if (light_idx < 0) continue;
        
        Light light;
        if(light_idx < light_count) light = lights[light_idx];
        else light.type = 4;

        LightSample light_sample;

        switch (light.type) {
            case 0: sample_point_light(light, hit_point, seed, light_sample); break;
            case 1: sample_area_light(light, hit_point, seed, light_sample); break;
            case 2: sample_spot_light(light, hit_point, seed, light_sample); break;
            case 3: sample_sphere_light(light, hit_point, seed, light_sample); break;
            case 4: sample_hdr(hit_point, seed, light_sample); break;
        }

        if (!light_sample.valid){
          continue;
        }

        // 阴影测试
        float light_distance = length(light_sample.position - hit_point);
        if (test_shadow(hit_point, normal, light_sample.direction, light_distance)) {
            continue;
        }
        
        // 计算 BRDF 贡献
        float3 wi = light_sample.direction;
        float ndotl = max(EPS, abs(dot(normal, wi)));
        float pdf;
        float3 bsdf_val = EvalBSDF(mat, wo, wi, normal, is_flip, pdf);
        // bsdf_val = max( min(bsdf_val, 50.0 * float3(1 / (ndotl + EPS), 1 / (ndotl + EPS), 1 / (ndotl + EPS))), float3(0.0, 0.0, 0.0));
        // pdf = max(pdf, 0.0);

        float3 light_contrib = light_sample.radiance * bsdf_val * ndotl;
        
        if (any(light_contrib > 0.0)) {
            // 计算光源采样PDF
            float light_select_pdf = light_power_weights[light_idx];
            float light_pdf = light_sample.pdf * light_select_pdf;
            

            if (light.type == 0 || light.type == 2) {
                // 点光源和聚光灯：直接加（delta光源）
                total_light += light_contrib / light_pdf;
            }
            else {
                // 面光源、球光源、HDR
                float mis_weight = mis_balance_weight(light_pdf, pdf);
                total_light += light_contrib * mis_weight / light_pdf;
            }
        }
    }
    return total_light / sample_times;
}

float IntersectRaySphere(inout RayDesc ray, inout float3 sphere_center, float sphere_radius) {
    float3 oc = ray.Origin - sphere_center;
    float a = dot(ray.Direction, ray.Direction);
    float b = dot(oc, ray.Direction);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    
    float delta = b * b - a * c;  // divide 2
    
    if (delta < 0.0) {
        return -1.0;
    }

    delta = sqrt(delta);
    float t = (-b - delta) / a;
    
    if (t < ray.TMin) {
        t = (-b + delta) / a;
    }

    return (t >= ray.TMin && t <= ray.TMax) ? t : -1.0;
    
}

void CheckHitSphereLight(inout uint light_count, inout RayDesc ray, inout RayPayload payload, inout uint closest_light_idx) {
    float closest_distance = ray.TMax;
    closest_light_idx = 0xFFFFFFFF;
    
    if (payload.hit) {
        closest_distance = length(payload.hit_point - ray.Origin);
    }
    
    for (int i = 0; i < light_count; ++i) {
        Light light = lights[i];
        
        if (light.type != 3 || !light.visible || !light.enabled) {
            continue;
        }
        
        float t = IntersectRaySphere(ray, light.position, light.radius);
        
        if (t > 0.0 && t < closest_distance) {
            closest_distance = t;
            closest_light_idx = i;
            
            payload.hit = true;
            payload.hit_point = ray.Origin + t * ray.Direction;
        }
    }
}
// ====================== 景深辅助函数 ====================

float2 concentric_sample_disk(inout uint seed) {
    float2 u = random2(seed) * 2.0 - 1.0;
    
    if (u.x == 0.0 && u.y == 0.0) {
        return float2(0.0, 0.0);
    }
    
    float theta, r;
    if (abs(u.x) > abs(u.y)) {
        r = u.x;
        theta = (PI / 4.0) * (u.y / u.x);
    } else {
        r = u.y;
        theta = (PI / 2.0) - (PI / 4.0) * (u.x / u.y);
    }
    
    return r * float2(cos(theta), sin(theta));
}

float3 compute_focal_point(float3 ray_origin, float3 ray_dir, float focal_distance) {
    // 计算光线与焦平面的交点
    // 假设焦平面垂直于相机前向方向
    float3 camera_forward = float3(0.0, 0.0, -1.0); // 相机空间前向
    float3 world_forward = mul((float3x3)camera_info.camera_to_world, camera_forward);
    world_forward = normalize(world_forward);
    
    // 焦平面方程: dot(p - (origin + world_forward * focal_distance), world_forward) = 0
    float3 focal_plane_origin = ray_origin + world_forward * focal_distance;
    float denom = dot(ray_dir, world_forward);
    
    if (abs(denom) > 1e-6) {
        float t = dot(focal_plane_origin - ray_origin, world_forward) / denom;
        return ray_origin + ray_dir * t;
    }
    
    // 平行于焦平面，使用近似值
    return ray_origin + ray_dir * focal_distance;
}


// 生成主射线与两个邻像素射线（用于差分）
// pixel: integer pixel coords (SV_DispatchThreadID or similar)
// pixel_sample_offset: subpixel jitter in [0,1) for anti-aliasing
// resolution: float2(width, height)
// out: main RayDesc (for TraceRay) and RayDifferential (to store in payload)
// ====================== 光线微分生成（修正版） ======================
void GenerateCameraRaysWithDifferentials(
    uint2 pixel, 
    float2 pixel_sample_offset, 
    float2 resolution,
    float2 lens_sample,  // 新增：透镜采样坐标（用于DOF）
    inout uint seed,     // 新增：随机数种子
    out RayDesc outRay, 
    out RayDifferential outDiffs)
{
    // 1. 计算像素中心与邻像素的NDC坐标
    float2 invRes = 1.0 / resolution;
    
    // 主像素
    float2 p_center = (float2(pixel) + pixel_sample_offset) * invRes;
    // 右邻像素 (x+1)
    float2 p_rx = (float2(pixel + uint2(1, 0)) + pixel_sample_offset) * invRes;
    // 上邻像素 (y+1)
    float2 p_ry = (float2(pixel + uint2(0, 1)) + pixel_sample_offset) * invRes;
    
    // 2. 转换到NDC [-1, 1]（注意y轴翻转）
    p_center = float2(p_center.x * 2.0 - 1.0, 1.0 - p_center.y * 2.0);
    p_rx = float2(p_rx.x * 2.0 - 1.0, 1.0 - p_rx.y * 2.0);
    p_ry = float2(p_ry.x * 2.0 - 1.0, 1.0 - p_ry.y * 2.0);
    
    // 3. 从screen到camera空间（使用近平面z=0）
    float4 cam_center = mul(camera_info.screen_to_camera, float4(p_center, 0.0, 1.0));
    float4 cam_rx = mul(camera_info.screen_to_camera, float4(p_rx, 0.0, 1.0));
    float4 cam_ry = mul(camera_info.screen_to_camera, float4(p_ry, 0.0, 1.0));
    
    cam_center.xyz /= cam_center.w;
    cam_rx.xyz /= cam_rx.w;
    cam_ry.xyz /= cam_ry.w;
    
    // 4. 处理DOF（薄透镜模型）
    float3 ray_origin_camera = float3(0.0, 0.0, 0.0);
    float3 rx_origin_camera = float3(0.0, 0.0, 0.0);
    float3 ry_origin_camera = float3(0.0, 0.0, 0.0);
    
    float3 ray_dir_camera = normalize(cam_center.xyz);
    float3 rx_dir_camera = normalize(cam_rx.xyz);
    float3 ry_dir_camera = normalize(cam_ry.xyz);
    
    if (camera_info.enable_depth_of_field && camera_info.lens_radius > 0.0) {
        // 计算焦平面上的点
        float t_focus = camera_info.focal_distance / max(-ray_dir_camera.z, EPS);
        float3 focal_point_camera = ray_dir_camera * t_focus;
        
        // 计算rx/ry射线在焦平面上的交点（近似处理）
        float t_focus_rx = camera_info.focal_distance / max(-rx_dir_camera.z, EPS);
        float t_focus_ry = camera_info.focal_distance / max(-ry_dir_camera.z, EPS);
        float3 focal_point_rx = rx_dir_camera * t_focus_rx;
        float3 focal_point_ry = ry_dir_camera * t_focus_ry;
        
        // 透镜采样（所有射线使用同一个透镜采样点，确保一致性）
        float2 lens_offset = lens_sample * camera_info.lens_radius;
        ray_origin_camera = float3(lens_offset, 0.0);
        rx_origin_camera = float3(lens_offset, 0.0);
        ry_origin_camera = float3(lens_offset, 0.0);
        
        // 更新光线方向（指向焦平面上的点）
        ray_dir_camera = normalize(focal_point_camera - ray_origin_camera);
        rx_dir_camera = normalize(focal_point_rx - rx_origin_camera);
        ry_dir_camera = normalize(focal_point_ry - ry_origin_camera);
    } else {
        // 无DOF，所有射线从相机原点出发
        ray_origin_camera = float3(0.0, 0.0, 0.0);
        rx_origin_camera = float3(0.0, 0.0, 0.0);
        ry_origin_camera = float3(0.0, 0.0, 0.0);
    }
    
    // 6. 转换到世界空间
    float4 world_origin = mul(camera_info.camera_to_world, float4(ray_origin_camera, 1.0));
    float4 world_rx_origin = mul(camera_info.camera_to_world, float4(rx_origin_camera, 1.0));
    float4 world_ry_origin = mul(camera_info.camera_to_world, float4(ry_origin_camera, 1.0));
    
    float4 world_dir = mul(camera_info.camera_to_world, float4(ray_dir_camera, 0.0));
    float4 world_rx_dir = mul(camera_info.camera_to_world, float4(rx_dir_camera, 0.0));
    float4 world_ry_dir = mul(camera_info.camera_to_world, float4(ry_dir_camera, 0.0));
    
    // 7. 输出主射线
    outRay.Origin = world_origin.xyz;
    outRay.Direction = normalize(world_dir.xyz);
    outRay.TMin = t_min;
    outRay.TMax = t_max;
    
    // 8. 输出光线微分
    outDiffs.hasDifferentials = true;
    outDiffs.rxOrigin = world_rx_origin.xyz;
    outDiffs.ryOrigin = world_ry_origin.xyz;
    outDiffs.rxDirection = normalize(world_rx_dir.xyz);
    outDiffs.ryDirection = normalize(world_ry_dir.xyz);
}


// ====================== 主渲染逻辑 ======================
[shader("raygeneration")]
void RayGenMain() {
    uint2 pixel_coords = DispatchRaysIndex().xy;
    uint seed = generate_seed(pixel_coords, render_setting.frame_count);
    // 生成透镜采样（用于DOF）
    float2 lens_sample = float2(0.0, 0.0);
    if (camera_info.enable_depth_of_field && camera_info.lens_radius > 0.0) {
        lens_sample = concentric_sample_disk(seed);
    }
    
    // 生成像素采样偏移（抗锯齿）
    float2 pixel_sample_offset = render_setting.enable_accumulation ? 
                                 random2(seed) : float2(0.5, 0.5);
    
    // 使用光线微分生成射线
    RayDesc ray;
    RayDifferential diffs;
    GenerateCameraRaysWithDifferentials(
        pixel_coords,
        pixel_sample_offset,
        render_setting.resolution,
        lens_sample,
        ray,
        diffs
    );
    
    // 应用相机运动模糊的时间偏移
    float time_offset = 0.0;
    if (camera_info.enable_motion_blur) {
        // 随机时间偏移（落在曝光时间范围内）
        time_offset = random(seed) * camera_info.exposure_time;

        // 1) 线性位移：主射线与邻射线原点应一致地平移
        float3 linear_move = camera_info.camera_linear_velocity * time_offset;
        ray.Origin += linear_move;
        // 注意：GenerateCameraRaysWithDifferentials 已经把 diffs.rxOrigin/ryOrigin 设置为 world-space，
        // 因此这里也需要同步平移它们，保持微分一致性（以便 closesthit 能正确计算 dpdx/dpdy）。
        diffs.rxOrigin += linear_move;
        diffs.ryOrigin += linear_move;

        // 2) 角速度（绕相机本地Z轴的旋转）：
        //    若角速度不为0，尽量对方向与原点都做相同的旋转（绕相机世界位置）。
        float angular_offset = camera_info.camera_angular_velocity * time_offset;
        if (abs(angular_offset) > 1e-8) {
            float cos_theta = cos(angular_offset);
            float sin_theta = sin(angular_offset);
            float3x3 rotation_matrix = float3x3(
                cos_theta, -sin_theta, 0.0,
                sin_theta,  cos_theta, 0.0,
                0.0,        0.0,       1.0
            );

            // 旋转方向向量（保持方向微分一致）
            ray.Direction = normalize(mul(rotation_matrix, ray.Direction));
            diffs.rxDirection = normalize(mul(rotation_matrix, diffs.rxDirection));
            diffs.ryDirection = normalize(mul(rotation_matrix, diffs.ryDirection));

            // 旋转原点：绕相机世界位置作为 pivot
            float3 cam_world_pos = mul(camera_info.camera_to_world, float4(0.0, 0.0, 0.0, 1.0)).xyz;
            ray.Origin = cam_world_pos + mul(rotation_matrix, ray.Origin - cam_world_pos);
            diffs.rxOrigin = cam_world_pos + mul(rotation_matrix, diffs.rxOrigin - cam_world_pos);
            diffs.ryOrigin = cam_world_pos + mul(rotation_matrix, diffs.ryOrigin - cam_world_pos);
        }

        // 3) 保守策略：启用 motion blur 时禁用基于射线微分的 adaptive mipmap
        //    （ClosestHitMain 已根据 payload.diffs.hasDifferentials 决定是否使用微分）
        diffs.hasDifferentials = false;
    }    

    // === 协方差初始化 ===
    float3x3 dirCov = (float3x3)0;
    bool hasDirCov = diffs.hasDifferentials;
    if (hasDirCov) {
        float3 v      = ray.Direction;
        float3 dv_dx  = diffs.rxDirection - v;
        float3 dv_dy  = diffs.ryDirection - v;

        // Cov_L = dv_dx dv_dx^T + dv_dy dv_dy^T
        dirCov[0][0] = dv_dx.x * dv_dx.x + dv_dy.x * dv_dy.x;
        dirCov[0][1] = dv_dx.x * dv_dx.y + dv_dy.x * dv_dy.y;
        dirCov[0][2] = dv_dx.x * dv_dx.z + dv_dy.x * dv_dy.z;

        dirCov[1][0] = dv_dx.y * dv_dx.x + dv_dy.y * dv_dy.x;
        dirCov[1][1] = dv_dx.y * dv_dx.y + dv_dy.y * dv_dy.y;
        dirCov[1][2] = dv_dx.y * dv_dx.z + dv_dy.y * dv_dy.z;

        dirCov[2][0] = dv_dx.z * dv_dx.x + dv_dy.z * dv_dy.x;
        dirCov[2][1] = dv_dx.z * dv_dx.y + dv_dy.z * dv_dy.y;
        dirCov[2][2] = dv_dx.z * dv_dx.z + dv_dy.z * dv_dy.z;
    }

    diffs.hasDifferentials = false; // 暂不启用
    hasDirCov = false;

    float3 color = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    entity_id_output[pixel_coords] = -1;

    float3 prev_hit_point;
    float prev_bsdf_pdf;
    bool prev_is_specular = false;

    uint light_count, light_idx;
    light_count = render_setting.light_count;


    for (int depth = 0; depth < min(render_setting.max_depth, MAX_DEPTH); ++depth) {

        RayPayload payload;
        payload.attribute = float4(-1.0,-1.0,-1.0,-1.0);
        payload.diffs = diffs;
        payload.dirCov   = dirCov;
        payload.hasDirCov = hasDirCov;

        TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
        // Traceray(ray, payload, 0);

        CheckHitSphereLight(light_count, ray, payload, light_idx);

        Material mat;

        if(light_idx == 0xFFFFFFFF){
            if (!payload.hit) {
                if(render_setting.skybox_texture_id_ >= 0){
                    light_idx = light_count;
                }
                else{ // no hdr
                    color += payload.color * throughput;
                    break;
                }
            }
            else{
                // 记录首次命中的实体ID
                if (depth == 0) {
                    entity_id_output[pixel_coords] = (int)payload.instance_id;
                }
                mat = materials[payload.instance_id];
                light_idx = mat.light_index;
            }
            mat.base_color = payload.color;
            if(payload.attribute.r>=-EPS)
            {
                mat.roughness = payload.attribute.r;
                mat.specular = payload.attribute.g;
                mat.metallic = payload.attribute.b;
                mat.roughness = payload.attribute.a;
            }
        }

        float3 wo = -ray.Direction;
        
        // 处理光源命中
        if (light_idx != 0xFFFFFFFF) {
            Light light;
            if(light_idx < light_count) light = lights[light_idx];
            else{
                light.enabled = true;
                light.type = 4;
                light.color = payload.color;
                light.intensity = 1.0;
            }

            if(light.enabled && (light.type != 1 || dot(wo, light.direction) > 0) ){ // 单向面光源

                if (depth == 0 || prev_is_specular || !render_setting.enable_accumulation) {
                    // 第一次直接击中光源或镜面反射击中光源
                    color += light.color * light.intensity * throughput;
                } else {
                    // BSDF采样击中光源，需要MIS
                    // if (light.type != 1 && light.type != 3) {
                    //     break;
                    // }
                                        
                    // 计算光源采样PDF
                    float light_pdf = 0.0;

                    if(light.type == 4){
                        float luminance = dot(payload.color, float3(0.2126, 0.7152, 0.0722));
                        light_pdf = luminance / hdr_cdf[2];

                    } else{
                        float3 to_light = payload.hit_point - prev_hit_point;
                        float distance = length(to_light);
                        float3 light_dir = to_light / distance;
                        float cos_theta_l = max(EPS, abs(dot(light_dir, light.direction)));
                        
                        if (light.type == 1) { // 面光源
                            float area = light.size.x * light.size.y;
                            light_pdf = (distance * distance) / (cos_theta_l * area);
                        } else if (light.type == 3) { // 球光源
                            float surface_area = 4.0 * PI * light.radius * light.radius;
                            light_pdf = (distance * distance) / (cos_theta_l * surface_area);
                        }
                    }
                    
                    float light_select_pdf = light_power_weights[light_idx];
                    light_pdf *= light_select_pdf;
                    float mis_weight = mis_balance_weight(prev_bsdf_pdf, light_pdf);
                    
                    // 应用MIS
                    color += light.color * light.intensity * throughput * mis_weight; // Here throughut equals to pre_throughput * prev_eval_brdf / prev_bsdf_pdf
                }
                break;
            }
        }

        prev_hit_point = payload.hit_point;

        // MIS直接光照
        if(render_setting.enable_accumulation){
            float3 direct_light = mis_direct_lighting(light_count, payload.hit_point, payload.normal, mat, wo, payload.is_filp, seed);
            color += direct_light * throughput;
        }
        
        // 处理自发光材质
        if (any(mat.emission > 0.0)) {
            color += mat.emission * throughput;
        }
        
        // 俄罗斯轮盘赌终止
        if (depth > 4) {
            float p_survive = min(f3_max(throughput), RR_THRESHOLD);
            if (random(seed) > p_survive) break;
            throughput /= p_survive;
        }
        
        // 采样下一跳方向（BSDF采样）
        float3 wi = float3(0.0, 0.0, 0.0);
        if (!SampleBSDF(mat, wo, payload.normal, wi, payload.is_filp, seed)){
            break;
        }

        // 更新：下一跳如 miss，用这次 BSDF 产生的 envLOD
        float pdf;
        float3 bsdf_val = EvalBSDF(mat, wo, wi, payload.normal, payload.is_filp, pdf);


        if(isinf(pdf) || isnan(pdf)){
          color = float3(1e9, 0.0, isnan(pdf)? 1e9: 0);
          break;
        }
        
        // 计算余弦项
        float cos_theta = dot(payload.normal, wi);
        if (cos_theta <= 0.0) {
            payload.normal = - payload.normal;
            cos_theta = - cos_theta;
        }

        bsdf_val *= cos_theta / pdf;

        // prevent extreme cases
        if (pdf < 1e-5 || f3_max(bsdf_val) > 1e3) {
            break;
        } 
        // if (pdf < 2e-6 || f3_max(bsdf_val) > 5e3) {
        //     break;
        // } 
        // 更新吞吐量
        throughput *= bsdf_val; // bsdf_val modified above
        // throughput = max(throughput, float3(0.0, 0.0, 0.0));


        if (mat.roughness > 0.6 && mat.metallic < 0.2 && mat.transparency < 0.2) {
            diffs.hasDifferentials = false;
            hasDirCov = false;
        }

        // 如果弹跳太深，也可以停掉
        if (depth > 4) {
            diffs.hasDifferentials = false;
            hasDirCov = false;            
        }

        if (diffs.hasDifferentials)
        {
            float3 v_in = ray.Direction;

            float3 dv_dx = diffs.rxDirection - v_in;
            float3 dv_dy = diffs.ryDirection - v_in;

            float3 dn_dx = payload.dndx;
            float3 dn_dy = payload.dndy;

            float NdotV = dot(payload.normal, wo);
            float NdotL = dot(payload.normal, wi);
            bool is_trans = (NdotV * NdotL) < 0.0;

            float3 dwo_dx = 0.0;
            float3 dwo_dy = 0.0;

            // === 3.1 入射方向协方差：直接用 payload.dirCov ===
            float3x3 Cov_in = payload.dirCov;
            bool covValid = payload.hasDirCov;

            // === 3.2 微法线协方差 Cov_h（与 anisotropic GGX 一致） ===
            float3 T, B;
            GetTangent(payload.normal, T, B);

            float alpha = mat.roughness * mat.roughness;
            float aspect = sqrt(max(1.0 - 0.9 * mat.anisotropic, EPS));
            float ax = max(EPS, alpha / aspect);
            float ay = max(EPS, alpha * aspect);

            float3x3 Cov_h = MicroNormalCovariance_Aniso(
                payload.normal, T, B, ax, ay
            );

            // === 3.3 选择反射/折射的 Jacobian ===
            float3x3 J_L_v;
            float3x3 J_L_n;

            if (!is_trans)
            {
                // 反射
                J_L_v = ReflectionJacobian_wrt_Dir(v_in, payload.normal);
                J_L_n = ReflectionJacobian_wrt_Normal(v_in, payload.normal);

                dwo_dx = float3(
                    dot(J_L_v[0], dv_dx) + dot(J_L_n[0], dn_dx),
                    dot(J_L_v[1], dv_dx) + dot(J_L_n[1], dn_dx),
                    dot(J_L_v[2], dv_dx) + dot(J_L_n[2], dn_dx)
                );
                dwo_dy = float3(
                    dot(J_L_v[0], dv_dy) + dot(J_L_n[0], dn_dy),
                    dot(J_L_v[1], dv_dy) + dot(J_L_n[1], dn_dy),
                    dot(J_L_v[2], dv_dy) + dot(J_L_n[2], dn_dy)
                );
            }
            else
            {
                // 折射
                float eta = (!payload.is_filp) ? (1.0 / mat.ior) : mat.ior;

                J_L_v = RefractionJacobian_wrt_Dir(v_in, payload.normal, eta);
                J_L_n = RefractionJacobian_wrt_Normal(v_in, payload.normal, eta);

                dwo_dx = float3(
                    dot(J_L_v[0], dv_dx) + dot(J_L_n[0], dn_dx),
                    dot(J_L_v[1], dv_dx) + dot(J_L_n[1], dn_dx),
                    dot(J_L_v[2], dv_dx) + dot(J_L_n[2], dn_dx)
                );
                dwo_dy = float3(
                    dot(J_L_v[0], dv_dy) + dot(J_L_n[0], dn_dy),
                    dot(J_L_v[1], dv_dy) + dot(J_L_n[1], dn_dy),
                    dot(J_L_v[2], dv_dy) + dot(J_L_n[2], dn_dy)
                );
            }

            // === 3.4 协方差递推：Cov_out = J_L_v Cov_in J_L_v^T + J_L_n Cov_h J_L_n^T ===
            if (covValid)
            {
                float3x3 Cov_geom  = PropagateNormalCovToDirectionCov(J_L_v, Cov_in);
                float3x3 Cov_micro = PropagateNormalCovToDirectionCov(J_L_n, Cov_h);

                dirCov = Cov_geom + Cov_micro;  // 真正的 Cov_out

                // 可选：trace 限制，防止爆炸
                float trace = dirCov[0][0] + dirCov[1][1] + dirCov[2][2];
                float maxTrace = 10.0; // 可按场景微调
                if (trace > maxTrace) {
                    float scale = maxTrace / max(trace, EPS);
                    dirCov *= scale;
                }
                hasDirCov = true;
            }

            // === 3.5 用 dwo_dx/dwo_dy 更新 ray differentials ===
            float3 wo_out = wi;

            diffs.rxDirection = normalize(wo_out + dwo_dx);
            diffs.ryDirection = normalize(wo_out + dwo_dy);

            diffs.rxOrigin = payload.hit_point + payload.dpdx;
            diffs.ryOrigin = payload.hit_point + payload.dpdy;
        }
        
        // 记录用于MIS的数据
        prev_bsdf_pdf = pdf;
        
        prev_is_specular = (mat.metallic > 0.9 || mat.clearcoat > 0.0); // simplified
        
        // 准备下一次光线
        ray.Origin = payload.hit_point + payload.normal * eps;
        ray.Direction = normalize(wi);
        ray.TMin = t_min;
        ray.TMax = t_max;
    }

    // color = max(color, float3(0.0, 0.0, 0.0));
    
    // 输出和累积
    output[pixel_coords] = float4(color, 1);
    
    if (render_setting.enable_accumulation) {
        float4 prev_color = accumulated_color[pixel_coords];
        int prev_samples = accumulated_samples[pixel_coords];
        accumulated_color[pixel_coords] = prev_color + float4(color, 1);
        accumulated_samples[pixel_coords] = prev_samples + 1;
    }
}

// // ====================== Miss着色器 ======================
[shader("miss")]
void MissMain(inout RayPayload payload) {
    payload.hit = false;
    payload.instance_id = 0xFFFFFFFF;
    
    if (render_setting.skybox_texture_id_ >= 0) {
        // 采样HDR天空盒
        float3 normalizedDir = normalize(WorldRayDirection());
        
        float phi = atan2(normalizedDir.z, normalizedDir.x);
        float theta = acos(clamp(normalizedDir.y, -1.0, 1.0));
        
        // 将球面坐标转换为UV坐标
        float u = (phi + PI) / (2.0 * PI);
        float v = theta / PI;
        
        // 确保UV在[0,1]范围内
        u = frac(u);
        v = clamp(v, 0.0, 1.0);
        
        float lod = 0.0;
        if (payload.hasDirCov) {
            float2x2 CovUV;
            DirectionCovToLatLongUVCov(normalizedDir, payload.dirCov, CovUV);

            float2 axis_len;
            float2x2 cov_texel;
            lod = ComputeEnvmapLOD_FromCovUV(
                render_setting.skybox_texture_id_, CovUV,
                axis_len, cov_texel
            );

            lod = clamp(lod, 0.0, 10.0);
        }

        payload.color = g_Textures[render_setting.skybox_texture_id_]
                            .SampleLevel(g_Sampler, float2(u, v), lod).rgb;
        // payload.color = g_Textures[render_setting.skybox_texture_id_].SampleLevel(g_Sampler, float2(u, v), 0).rgb;
    } else {
        // 简化的天空渐变（原有的）
        float t = 0.5 * (normalize(WorldRayDirection()).y + 1.0);
        payload.color = lerp(float3(0.5, 0.7, 1.0), float3(1.0, 1.0, 1.0), t) * 0.8;
    }
}


// ========================== Adaptive mipmap 法线差分 ====================================

// 计算用于 SampleLevel 的 LOD
float ComputeTextureLOD(Texture2D<float4> tex,
                        float dudx, float dudy, float dvdx, float dvdy)
{
    uint w, h;
    tex.GetDimensions(w, h);
    // 把 uv 导数转换到 texel 空间
    float2 dx = float2(dudx * (float)w, dvdx * (float)h);
    float2 dy = float2(dudy * (float)w, dvdy * (float)h);
    float rho2 = max(dot(dx, dx), dot(dy, dy));
    // 防止 log2(0)
    rho2 = max(rho2, 1e-8);
    float lod = 0.5 * log2(rho2);
    if (lod < 0.0) lod = 0.0;
    return lod;
}

// --- 1.1 计算三角形切线/副切线（per-triangle, constant） ---
// 返回 true 表示成功（UV 矩阵非退化），并输出 tangent, bitangent（未归一化后可 normalize）
bool ComputeTriangleTangentBasis(
    float3 v0, float3 v1, float3 v2,
    float2 uv0, float2 uv1, float2 uv2,
    out float3 tangent, out float3 bitangent)
{
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float du1 = uv1.x - uv0.x;
    float du2 = uv2.x - uv0.x;
    float dv1 = uv1.y - uv0.y;
    float dv2 = uv2.y - uv0.y;

    float det = du1 * dv2 - du2 * dv1;
    float tol = 1e-8;
    if (abs(det) < tol) {
        tangent = float3(0,0,0);
        bitangent = float3(0,0,0);
        return false;
    }
    float invDet = 1.0 / det;
    tangent = ( E1 * dv2 - E2 * dv1 ) * invDet;
    bitangent = ( -E1 * du2 + E2 * du1 ) * invDet;
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);
    return true;
}

// --- 1.2 顶点法线插值导数（barycentric 导数法） ---
// ---------------------------------------------------------------------------
// 推荐实现：ComputeSmoothNormalDerivatives_Triangle
// 说明：
//  - 采用你约定的重心对应关系：u = w0 = attr.barycentrics.x 对应 v1，
//    v = w1 = attr.barycentrics.y 对应 v2，v0 权重为 (1-u-v)。
//  - 需要保证传入的 E1,E2,dpdx,dpdy,n0,n1,n2 都处于同一坐标系（通常 world space）。
//  - 依赖 SolveForCoeffs3x2(E1,E2, dp) -> float2(a,b)（得到 a=∂u/∂x, b=∂v/∂x）。
// ---------------------------------------------------------------------------
void ComputeSmoothNormalDerivatives_Triangle(
    float3 n0, float3 n1, float3 n2,    // 顶点法线 (同一坐标系)
    float3 E1, float3 E2,              // E1 = v1 - v0, E2 = v2 - v0 (同一坐标系)
    float2 bary,                       // bary.x = w0 (=u, 对应 v1), bary.y = w1 (=v, 对应 v2)
    float3 dpdx, float3 dpdy,          // 屏幕空间到三维空间的微分 (同一坐标系)
    out float3 dndx, out float3 dndy,  // 输出：平滑法线在屏幕 x,y 的导数（world-space）
    out float3 n_shading)              // 输出：插值后（未受 normal-map 影响）的 shading normal（已归一化）
{
    // 1) 计算 shading normal（按你的重心约定）
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;            // 对应 v0 的权重
    // 线性插值并归一化
    float3 n_interp = u * n1 + v * n2 + w * n0;
    float lenN = length(n_interp);
    if (lenN < EPS) {
        // 极端退化：使用面法线近似（E1 x E2）
        n_shading = normalize(cross(E1, E2));
        if (length(n_shading) < EPS) {
            // 彻底退化：任取一个轴
            n_shading = float3(0.0, 0.0, 1.0);
        }
    } else {
        n_shading = normalize(n_interp);
    }

    // 2) 计算法线对局部参数 u,v 的偏导：对线性重心插值有解析表达
    //    ∂n/∂u = n1 - n0 ;  ∂n/∂v = n2 - n0
    float3 dn_du = n1 - n0;
    float3 dn_dv = n2 - n0;

    // 3) 使用 SolveForCoeffs3x2 求解 dpdx = a*E1 + b*E2 （a = ∂u/∂x, b = ∂v/∂x）
    //    并同理求出 ∂u/∂y, ∂v/∂y
    float2 ab_dx = SolveForCoeffs3x2(E1, E2, dpdx); // returns (a_dx, b_dx)
    float2 ab_dy = SolveForCoeffs3x2(E1, E2, dpdy); // returns (a_dy, b_dy)

    // 4) 使用链式法则得到 dndx, dndy
    //    dndx = (∂n/∂u) * (∂u/∂x) + (∂n/∂v) * (∂v/∂x)
    dndx = dn_du * ab_dx.x + dn_dv * ab_dx.y;
    dndy = dn_du * ab_dy.x + dn_dv * ab_dy.y;

    // 5) 数值保护（可选）：如果 dp 很小导致 ab_* 都几乎为 0，则保持 dndx/dndy 为 0
    //    （上述计算已自然返回 0，若需要可在此 clamp）
}

void ComputeNormalMapDerivatives_WithCustomLOD(
    Texture2D<float4> tex, SamplerState samp,
    float2 uv,
    float dudx, float dudy, float dvdx, float dvdy,
    float custom_lod, // 自定义LOD值
    out float3 n_t, out float3 dntdx, out float3 dntdy)
{
    // 1) 采样中心像素
    float4 c = tex.SampleLevel(samp, uv, custom_lod);
    float3 sC = c.xyz;
    float3 nc = sC * 2.0 - 1.0;
    n_t = normalize(nc);

    // 2) 获取当前mip级别的纹理尺寸
    uint w, h;
    tex.GetDimensions(w, h); 
    
    // 3) 根据自定义LOD计算相邻像素的偏移
    float mip_scale = exp2(custom_lod);
    float2 texel = 1.0 / float2(max(1u, w), max(1u, h)) * mip_scale;
    
    // 4) 采样相邻像素用于中心差分
    float4 su = tex.SampleLevel(samp, uv + float2(texel.x, 0.0), custom_lod);
    float4 su_m = tex.SampleLevel(samp, uv - float2(texel.x, 0.0), custom_lod);
    float4 sv = tex.SampleLevel(samp, uv + float2(0.0, texel.y), custom_lod);
    float4 sv_m = tex.SampleLevel(samp, uv - float2(0.0, texel.y), custom_lod);

    // 5) 计算导数
    float3 dn_du = (su.xyz - su_m.xyz) / texel.x;
    float3 dn_dv = (sv.xyz - sv_m.xyz) / texel.y;

    // 6) 链式法则转换到屏幕空间
    dntdx = dn_du * dudx + dn_dv * dvdx;
    dntdy = dn_du * dudy + dn_dv * dvdy;
}

// 综合：计算最终 world-space 法线导数并写入 payload.dndx/dndy
// 参数：
//   v0,v1,v2: triangle positions
//   n0,n1,n2: vertex normals (shading normals)
//   uv0,uv1,uv2: vertex uvs
//   bary: (attr.barycentrics.x, attr.barycentrics.y)
//   normal_tex_id: -1 if none
//   payload: inout RayPayload (包含 dpdx/dpdy, dudx/dudy,dvdx/dvdy)
// ---------- 替换(并扩展签名): UpdateNormalDerivativesAndNormalMap ----------
// 说明：新增参数 uv_hit —— hit 点的插值 UV（使用 closesthit 里计算好的 object_texcoord）
void UpdateNormalDerivativesAndNormalMap(
    float3 v0, float3 v1, float3 v2,
    float3 n0, float3 n1, float3 n2,
    float2 uv0, float2 uv1, float2 uv2,
    float2 uv_hit,
    float2 bary,
    int normal_tex_id,
    float custom_lod, // 新增：自定义LOD值
    inout RayPayload payload)
{
    // 1) 三角形边
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;

    // 2) 计算顶点法线插值导数（world space）以及 shading normal（作为基准）
    float3 dndx_vert = float3(0,0,0);
    float3 dndy_vert = float3(0,0,0);
    float3 n_shading = float3(0,0,1);

    ComputeSmoothNormalDerivatives_Triangle(n0, n1, n2, E1, E2, bary, payload.dpdx, payload.dpdy, dndx_vert, dndy_vert, n_shading);

    // 3) 计算切线 / bitangent（triangle-level）
    float3 triT, triB;
    bool okTB = ComputeTriangleTangentBasis(v0, v1, v2, uv0, uv1, uv2, triT, triB);
    if (!okTB) {
        // 如果失败，基于 shading normal 构造任意正交基
        GetTangent(n_shading, triT, triB);
    } else {
        // 保证正交化并归一化 (Gram-Schmidt)
        triT = normalize(triT - n_shading * dot(n_shading, triT));
        triB = normalize(cross(n_shading, triT));
    }

    // 4) 如果存在 normal map，则采样 normal map 并计算其切线空间导数
    float3 dndx_map = float3(0,0,0);
    float3 dndy_map = float3(0,0,0);
    float3 final_world_normal = normalize(n_shading);

    if (normal_tex_id >= 0) {
        // 4.1 使用传入的custom_lod而不是内部计算
        Texture2D<float4> normal_tex = g_Textures[normal_tex_id];
        
        // 修改ComputeNormalMapDerivatives以支持自定义LOD
        float3 n_t; float3 dntdx_t; float3 dntdy_t;
        ComputeNormalMapDerivatives_WithCustomLOD( // 需要创建新函数
            normal_tex, g_Sampler, uv_hit,
            payload.dudx, payload.dudy, payload.dvdx, payload.dvdy,
            custom_lod, // 使用自定义LOD
            n_t, dntdx_t, dntdy_t);

        // 4.2 切线空间 (tangent-space) -> world-space 的变换
        // world normal from map:
        final_world_normal = normalize(triT * n_t.x + triB * n_t.y + n_shading * n_t.z);

        // 4.3 切线空间导数转换到 world-space
        // 近似：假定 triT, triB, n_shading 在三角片上近似常量（即 dT/dx≈0），因此：
        dndx_map = triT * dntdx_t.x + triB * dntdx_t.y + n_shading * dntdx_t.z;
        dndy_map = triT * dntdy_t.x + triB * dntdy_t.y + n_shading * dntdy_t.z;

        // 注：若需要更高精度，可以在三角片级别估计 triT/d triB/d（复杂度增）
    }

    // 5) 合并顶点插值导数与 normal-map 导数
    payload.dndx = dndx_vert + dndx_map;
    payload.dndy = dndy_vert + dndy_map;

    // 6) 写回最终 world-space normal（使用 normal map 优先）
    payload.normal = normalize(final_world_normal);
}

[shader("closesthit")]
void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // 标记与基本信息
    payload.hit = true;
    payload.instance_id = InstanceID();
    uint primitive_id = PrimitiveIndex();

    // world-space 射线和 t
    float3 ray_origin = WorldRayOrigin();
    float3 ray_direction = WorldRayDirection();
    float hit_distance = RayTCurrent();
    payload.hit_point = ray_origin + hit_distance * ray_direction;

    // 读取三角形顶点索引与原始顶点属性（object space）
    GeometryDescriptor geo_desc = geometry_descriptors[payload.instance_id];
    uint index_offset = geo_desc.index_offset + primitive_id * 3;
    uint i0 = indices[index_offset];
    uint i1 = indices[index_offset + 1];
    uint i2 = indices[index_offset + 2];

    VertexInfo v0 = vertices[geo_desc.vertex_offset + i0];
    VertexInfo v1 = vertices[geo_desc.vertex_offset + i1];
    VertexInfo v2 = vertices[geo_desc.vertex_offset + i2];

    // 重心按你给定的约定
    float w0 = attr.barycentrics.x; // weight for v1
    float w1 = attr.barycentrics.y; // weight for v2
    float w2 = 1.0 - w0 - w1;       // weight for v0

    // 插值 UV 按你的约定
    float2 uv_hit = w0 * v1.texcoord + w1 * v2.texcoord + w2 * v0.texcoord;

    // 将顶点位置与法线转换到 world-space —— 这一点很关键，保证 E1/E2 与 dpdx/dpdy 同坐标系
    float4 p0_w4 = mul(ObjectToWorld4x3(), v0.position);
    float4 p1_w4 = mul(ObjectToWorld4x3(), v1.position);
    float4 p2_w4 = mul(ObjectToWorld4x3(), v2.position);
    float3 p0_w = p0_w4.xyz;
    float3 p1_w = p1_w4.xyz;
    float3 p2_w = p2_w4.xyz;

    // 顶点法线从 object -> world（使用 3x3 部分）
    float3x3 objToWorld3 = (float3x3)ObjectToWorld4x3();
    float3 n0_w = normalize(mul(objToWorld3, v0.normal));
    float3 n1_w = normalize(mul(objToWorld3, v1.normal));
    float3 n2_w = normalize(mul(objToWorld3, v2.normal));

    // 三角形边（world-space）
    float3 E1_w = p1_w - p0_w; // corresponds to v1 - v0 (matches bary w0 mapping)
    float3 E2_w = p2_w - p0_w; // corresponds to v2 - v0

    // 计算位置微分 dpdx / dpdy（使用 ray differential 近似 p_rx = rxOrigin + t * rxDir，一个简单的近似）
    if (payload.diffs.hasDifferentials) {
        float3 hit_rx = payload.diffs.rxOrigin + hit_distance * payload.diffs.rxDirection;
        float3 hit_ry = payload.diffs.ryOrigin + hit_distance * payload.diffs.ryDirection;
        payload.dpdx = hit_rx - payload.hit_point;
        payload.dpdy = hit_ry - payload.hit_point;
    } else {
        payload.dpdx = float3(0.0, 0.0, 0.0);
        payload.dpdy = float3(0.0, 0.0, 0.0);
    }

    // 通过 SolveForCoeffs3x2 解出 ∂u/∂x, ∂v/∂x 与 ∂u/∂y, ∂v/∂y
    float2 ab_dx = SolveForCoeffs3x2(E1_w, E2_w, payload.dpdx); // a_dx = ∂u/∂x, b_dx = ∂v/∂x
    float2 ab_dy = SolveForCoeffs3x2(E1_w, E2_w, payload.dpdy); // a_dy = ∂u/∂y, b_dy = ∂v/∂y

    // 计算 uv 导数（注意 uv10 = uv1 - uv0, uv20 = uv2 - uv0；与重心 u=w0 对应 v1 保持一致）
    float2 uv10 = v1.texcoord - v0.texcoord;
    float2 uv20 = v2.texcoord - v0.texcoord;
    payload.dudx = uv10.x * ab_dx.x + uv20.x * ab_dx.y;
    payload.dvdx = uv10.y * ab_dx.x + uv20.y * ab_dx.y;
    payload.dudy = uv10.x * ab_dy.x + uv20.x * ab_dy.y;
    payload.dvdy = uv10.y * ab_dy.x + uv20.y * ab_dy.y;

    // 计算各纹理的 LOD 并以 LOD 采样 attribute / base color
    Material mat = materials[InstanceID()];
    int attribute_id = mat.attribute_tex_id;
    int texture_id = mat.texture_id;
    int normal_id = mat.normal_tex_id;
    

    if (attribute_id >= 0) {
        float lod_attr = ComputeTextureLOD(g_Textures[attribute_id], payload.dudx, payload.dudy, payload.dvdx, payload.dvdy);
        payload.attribute = g_Textures[attribute_id].SampleLevel(g_Sampler, uv_hit, lod_attr).rgba;
    } else {
        payload.attribute = float4(-1.0, -1.0, -1.0, -1.0);
    }

    if (texture_id >= 0) {
        float lod_col = ComputeTextureLOD(g_Textures[texture_id], payload.dudx, payload.dudy, payload.dvdx, payload.dvdy);
        payload.color = g_Textures[texture_id].SampleLevel(g_Sampler, uv_hit, lod_col).rgb;
    } else {
        payload.color = mat.base_color;
    }

    uint normal_tex_width = 0;
    uint normal_tex_height = 0;
    if (normal_id >= 0) {
        g_Textures[normal_id].GetDimensions(normal_tex_width, normal_tex_height);
    }

    // -------------------------------------------------
    // Adaptive normal map mip LOD
    // -------------------------------------------------
    float lod_normal = 0.0;

    if (payload.diffs.hasDifferentials && normal_id >= 0)
    {
        // A. surface-driven LOD（基于屏幕空间导数）
        lod_normal = ComputeTextureLOD(
            g_Textures[normal_id],
            payload.dudx, payload.dudy, payload.dvdx, payload.dvdy
        );
        // lod_normal = min(lod_normal, 10.0f); // 限制最大10级mip
    }
    // 计算并更新法线与法线导数（把 world-space 顶点位置与 world-space 顶点法线传入）
    // 注意：UpdateNormalDerivativesAndNormalMap 要求 v0..v2 与 n0..n2 在同一坐标系（我们已把它们变为 world-space）
    float2 bary = float2(w0, w1);
    UpdateNormalDerivativesAndNormalMap(
        p0_w, p1_w, p2_w,
        n0_w, n1_w, n2_w,
        v0.texcoord, v1.texcoord, v2.texcoord,
        uv_hit,
        bary,
        normal_id,
        lod_normal,
        payload);

    // 最终确保法线朝向与原射线一致（与原行为一致）
    payload.is_filp = false;
    if (dot(payload.normal, ray_direction) > 0.0) {
        payload.normal = -payload.normal;
        payload.is_filp = true;
    }
}