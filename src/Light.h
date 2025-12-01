#pragma once
#include "long_march.h"

struct Light {
    glm::vec3 position; //光源中心
    glm::vec3 color;    //光源颜色
    float intensity;    //光源强度
    float radius;       //点光源半径
    glm::vec3 normal;   //面光源法线
    glm::vec2 size;     //面光源尺寸(矩形-长宽)
    int type;           //0=点,1=面

    Light():position(0.0f),color(1.0f),intensity(1.0f),raidus(0.1f),normal(0.0f,-1.0f,0.0f),size(1.0f,1.0f),type(0){}
    
    static Light CreatePointLight(const glm::vec3& pos,const glm::vec3& col=glm::vec3(1.0f),float intens=1.0f,float rad=0.1f)
    {
        Light light;
        light.position=pos;
        light.color=col;
        light.intensity=intens;
        light.radius=rad;
        light.type=0;
        return light;
    }
    
    static Light CreateAreaLight(const glm::vec3& pos,const glm::vec3& norm,const glm::vec2& sz,const glm::vec3& col=glm::vec3(1.0f),float intens=1.0f)
    {
        Light light;
        light.position=pos;
        light.color=col;
        light.intensity=intens;
        light.normal=glm::normalize(norm);
        light.size=sz;
        light.type=1;
        return light;
    }
};