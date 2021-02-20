from OpenGL.GL import *
import pygame
from my_engine.mesh import Mesh
from my_engine.material import Material
import numpy as np
from pyo import *
from my_engine.texture import Texture

pygame.init()
pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

depth_buff = glGenRenderbuffers(1)
glBindRenderbuffer(GL_RENDERBUFFER, depth_buff)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1280, 720)

mat = Material("""
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform float time;
uniform float noise_intensity;

out vec2 uv;
out float intensity;
out float nois;

const float PHI = 1.61803398874989484820459; // Φ = Golden Ratio 

float gold_noise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}


void main()
{   
    // a_texture + 
    //vec2 nois = vec2(a_position.x + (a_position.z * 0.5f), a_position.y + (a_position.z * 0.5f));
    //vec3 rgba = vec3(gold_noise(nois, fract(time)+1.0f), // r
    //            gold_noise(nois, fract(time)+2.0f), // g
    //            gold_noise(nois, fract(time)+3.0f));
    //gl_Position = projection_matrix * view_matrix * model_matrix * vec4(a_position + (rgba * noise_intensity) , 1.0);
    //intensity = 10.0f/(1.0f + length(gl_Position.xyz));
    gl_Position = vec4(a_position, 1.0);
    nois = gold_noise(uv, time);
    uv = a_texture;
}
""", """
# version 330

in vec2 uv;
in float intensity;
in float nois;
out vec4 out_color;

uniform sampler2D texture;
//uniform sampler2D texture_kiti;
//uniform sampler2D texture_kat;

mat3 sx = mat3( 
    1.0, 2.0, 1.0, 
    0.0, 0.0, 0.0, 
   -1.0, -2.0, -1.0 
);
mat3 sy = mat3( 
    1.0, 0.0, -1.0, 
    2.0, 0.0, -2.0, 
    1.0, 0.0, -1.0 
);

void main()
{
    //out_color = vec4(texture(texture, vec2(uv.x, -uv.y) + vec2(0.005, 0.0)).x, texture(texture, vec2(uv.x, -uv.y) + vec2(0.0, 0.005)).y, texture(texture, vec2(uv.x, -uv.y) + vec2(0.005, 0.005)).z, texture(texture, vec2(uv.x, -uv.y)).w); 
    out_color = texture(texture, vec2(uv.x, -uv.y));
    mat3 I;
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            vec3 sample  = texelFetch(texture, ivec2(uv.x * 1280.0, (1-uv.y) * 720.0) + ivec2(i-1,j-1), 0 ).xyz;
            I[i][j] = length(sample); 
        }
    }
    float gx = dot(sx[0], I[0]) + dot(sx[1], I[1]) + dot(sx[2], I[2]); 
    float gy = dot(sy[0], I[0]) + dot(sy[1], I[1]) + dot(sy[2], I[2]);
    float res = sqrt(gx * gx + gy * gy);
    //out_color.xyz = vec3(res);
    float is_bigger = float(res > 0.9);
    out_color.xyz -= vec3(is_bigger);
    //out_color.xyz = is_bigger * (out_color.xyz / res) + (1.0 - is_bigger) * out_color.xyz;
    //if (uv.x < 0.5){
    //    out_color *= texture(texture_kiti, uv);
    //}
    //else{
    //    out_color *= texture(texture_kat, uv);
    //}
}
""")

mat1 = Material("""
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 instance_offset;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform float time;
uniform float noise_intensity;

out vec2 uv;
out float intensity;

const float PHI = 1.61803398874989484820459; // Φ = Golden Ratio 

vec2 hash( vec2 p ) // replace this by something better
{
    p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

    vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
    vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ),0.0);
    vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3(70.0));
}

float fractal_noise(vec2 p){
    float f = 0.0;
    p *= 5.0;
    mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
    f  = 0.5000*noise( p ); p = m*p;
    f += 0.2500*noise( p ); p = m*p;
    f += 0.1250*noise( p ); p = m*p;
    f += 0.0625*noise( p ); p = m*p;
    f = 0.5 + 0.5*f;
    f *= smoothstep( 0.0, 0.005, abs(p.x-0.6) );
    return f;
}

float gold_noise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}


void main()
{   
    // a_texture + 
    vec2 nois = vec2(a_position.x + (a_position.z * 0.5f), a_position.y + (a_position.z * 0.5f));
    vec3 rgba = vec3(gold_noise(nois, fract(time)+1.0f), // r
                gold_noise(nois, fract(time)+2.0f), // g
                gold_noise(nois, fract(time)+3.0f));
    
    vec4 model_offset = model_matrix * vec4(instance_offset, 1.0); 
    model_offset.y += (fractal_noise(model_offset.xz / 50.0) - 0.5)*3.0;
    gl_Position = projection_matrix * view_matrix * ((model_matrix * vec4(a_position + (rgba * noise_intensity), 1.0)) + model_offset);
    intensity = 10.0f/(1.0f + length(gl_Position.xyz));
    uv = a_texture;
}
""", """
# version 330

in vec2 uv;
in float intensity;
out vec4 out_color;

uniform sampler2D texture;

void main()
{
    out_color = texture(texture, uv); // * vec4(v_color, 1.0f);
    // out_color = vec4(vec3(out_color) * intensity, out_color.w);
    //out_color = vec4(vec3(out_color) * length(out_color - texture(texture, uv + vec2(0.1f, -0.1f))), out_color.w);
}
""", instanced_attribs=['instance_offset'])

plane_mat = Material("""
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform float time;
uniform float noise_intensity;

out vec2 uv;
out float intensity;

const float PHI = 1.61803398874989484820459; // Φ = Golden Ratio 

vec2 hash( vec2 p ) // replace this by something better
{
    p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

    vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
    vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ),0.0);
    vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3(70.0));
}

float fractal_noise(vec2 p){
    float f = 0.0;
    p *= 5.0;
    mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
    f  = 0.5000*noise( p ); p = m*p;
    f += 0.2500*noise( p ); p = m*p;
    f += 0.1250*noise( p ); p = m*p;
    f += 0.0625*noise( p ); p = m*p;
    f = 0.5 + 0.5*f;
    f *= smoothstep( 0.0, 0.005, abs(p.x-0.6) );
    return f;
}

float gold_noise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}


void main()
{   
    // a_texture + 
    vec2 nois = vec2(a_position.x + (a_position.z * 0.5f), a_position.y + (a_position.z * 0.5f));
    vec3 rgba = vec3(gold_noise(nois, fract(time)+1.0f), // r
                gold_noise(nois, fract(time)+2.0f), // g
                gold_noise(nois, fract(time)+3.0f));
    vec4 model_pos = model_matrix * vec4(a_position + (rgba * noise_intensity / 5.0), 1.0);
    model_pos.y += (fractal_noise(model_pos.xz / 100.0) - 0.5)*5.0;
    gl_Position = projection_matrix * view_matrix * model_pos;
    float lngth = length(gl_Position.xyz);
    intensity = (100.0f - model_pos.y)/(1.0f + lngth * lngth);
    uv = a_texture;
}
""", """
# version 330

in vec2 uv;
in float intensity;
out vec4 out_color;

uniform sampler2D texture;

void main()
{
    out_color = texture(texture, uv * 2.0); // * vec4(v_color, 1.0f);
    out_color.xyz /= (gl_FragCoord.z/gl_FragCoord.w + 1.0)/10.0;
    //out_color.xyz *= intensity;
    // out_color = vec4(vec3(out_color) * intensity, out_color.w);
    //out_color = vec4(vec3(out_color) * length(out_color - texture(texture, uv + vec2(0.1f, -0.1f))), out_color.w);
}
""")
import time
import math
import random
mesh = Mesh()
mesh.load_from_file("kulken.obj")
mesh.vertices_mapping = 'a_position'
mesh.point_data_mapping['uvs'] = 'a_texture'
mesh.uniform_data['time'] = lambda: (time.time() / random.randint(1, 1000)) % 1
mesh.uniform_data['noise_intensity'] = lambda: (math.sin(time.time() * math.pi * 2) + 1) / 2
del mesh.point_data_mapping['normals']
del mesh.point_data['normals']

background = Mesh()
background.load_from_file("screen.obj")
background.vertices_mapping = 'a_position'
background.point_data_mapping['uvs'] = 'a_texture'
background.uniform_data['time'] = 0 #lambda: (time.time() / random.randint(1, 1000)) % 1
background.uniform_data['noise_intensity'] = 0 # lambda: (math.sin(time.time() * math.pi * 2) + 1) / 2
del background.point_data_mapping['normals']
del background.point_data['normals']
# del background.point_data['a_position']

kotmesh = Mesh()
kotmesh.load_from_file("lowpoly_tree.obj")
kotmesh.vertices_mapping = 'a_position'
kotmesh.point_data_mapping['uvs'] = 'a_texture'
kotmesh.uniform_data['time'] = lambda: (time.time() / random.randint(1, 1000)) % 1
kotmesh.uniform_data['noise_intensity'] = lambda: (math.sin(time.time() * math.pi * 2) + 1) / 2
kotmesh.instanced_point_data['instance_offset'] = np.array([[random.uniform(-20, 20), 0, random.uniform(-20, 20)] for i in range(1000)])
kotmesh.point_data_mapping['instance_offset'] = 'instance_offset'
del kotmesh.point_data_mapping['normals']
del kotmesh.point_data['normals']

mesh1 = Mesh()
mesh1.load_from_file("plane.obj")
mesh1.vertices_mapping = 'a_position'
mesh1.point_data_mapping['uvs'] = 'a_texture'
mesh1.uniform_data['time'] = lambda: (time.time() / random.randint(1, 1000)) % 1

mesh1_noise_intens = 0
def nois_intens():
    global mesh1_noise_intens
    mesh1_noise_intens = max(0, mesh1_noise_intens - 0.1)
    return mesh1_noise_intens
mesh1.uniform_data['noise_intensity'] = nois_intens
del mesh1.point_data_mapping['normals']
del mesh1.point_data['normals']
# dat = mesh.get_data_positioned_to_material(mat)
from my_engine.game_object import GameObject

obj = GameObject()
obj.add_component(kotmesh)#mesh)
parent = GameObject()
obj.add_component(mat1)
# parent.add_child(obj)
from my_engine.camera import Camera


backtex = Texture()
backtex.load_from_file('sky_tex.jpg')
background_obj = GameObject(name='BACKGROUND')
background_obj.add_component(background)
background_obj.scale = np.ones(3)
# background_obj.rotation = (0, 0, math.pi/2)
# background_obj.add_component(backtex)
background_obj.add_component(mat)
parent.add_child(background_obj)

cam_obj = GameObject()
cam = Camera(cam_obj)
cam.objects = [background_obj]
cam_obj.translation = (0, 2, 20)

FBO = glGenFramebuffers(1)

framebuffer_texture = Texture(texture=glGenTextures(1), dont_send=True)
framebuffer_texture.bind_texture()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1280, 720, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
framebuffer_texture.bind_default_texture()

depth_buff = glGenRenderbuffers(1)
glBindRenderbuffer(GL_RENDERBUFFER, depth_buff)
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1280, 720)

framebuffer_texture.bind_to_frame_depth_buffer(FBO, depth_buff)

background.uniform_data['texture'] = framebuffer_texture

# kitittex = Texture(texture=glGenTextures(1))
# kitittex.load_from_file('kiti.png')
# kitittex.send_texture()
# background.uniform_data['texture_kiti'] = kitittex
#
# kitittex1 = Texture(texture=glGenTextures(1))
# kitittex1.load_from_file('cat.png')
# kitittex1.send_texture()
# background.uniform_data['texture_kat'] = kitittex1


# background_obj.add_component(framebuffer_texture)

cam_obj1 = GameObject(name='cam1')
cam_obj1.translation = (0, 2, 20)
cam1 = Camera(cam_obj1, frame_buffer=FBO, depth_buffer=depth_buff)
from my_engine.renderer import Renderer

kotek = GameObject()
cam_obj.add_child(kotek)
kotek.add_component(kotmesh)
kottex = Texture()
kottex.load_from_file('Cat_diffuse.jpg')
kotek.add_component(kottex)
kotek.translation = (0,-2,0)
# kotek.rotation = (random.uniform())
kotek.scale = (3,3,3)
# parent.add_child(cam_obj)

# parent.add_child(kotek)
render = Renderer(parent, [cam1, cam])
obj1 = GameObject()
obj1.add_component(mesh1)
obj1.add_component(plane_mat)
obj1.scale = tuple(np.array(obj1.scale) / 5)
obj.scale = tuple(np.array(obj.scale) * 0.2)

cam1.objects = [obj1]
cam.start()
cam1.start()


parent.add_child(obj1)
obj.translation = (0., 0, 0)
# obj1.translation = (0., 0., -10.)
obj2 = GameObject()
obj2.add_component(mesh)
obj2.add_component(mat)
# obj1.add_child(obj2)
obj2.scale = tuple(np.array(obj2.scale) * 3)
obj2.translation = (1., -1., -1.)


tex = Texture()
tex.load_from_file('cat.png')
tex1 = Texture()
tex1.load_from_file('sea.jpg')
obj.add_component(kottex)
obj1.add_component(tex1)
obj2.add_component(tex)
render.start()
# render.update()
running = True
import time
from threading import Thread

t0 = time.time()
rate = 0

def funn():
    global running
    global t0
    global mesh1_noise_intens
    while running:
        rate = time.time() - t0
        t0 = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # obj1.translation -= np.array([0., 0., rate]) #obj1.translation[0], obj1.translation[1], obj1.translation[2] - rate)
        # obj1.rotation -= np.array([0., 0., rate])
        x = np.array([random.uniform(-0.1, 0.5), 0 ,random.uniform(-0.1, 0.5)]) / 10
        obj1.translation += x
        cam_obj.translation += x
        cam_obj1.translation = cam_obj.translation
        cam_obj1.rotation = cam_obj.rotation
        background_obj.translation += x
        # random_rot = np.random.rand(3)
        # obj.rotation -= random_rot * rate / np.sum(random_rot)
        render.update()
        pygame.display.flip()
        render.update()
        # print(1/max(rate, 0.00001))
        mesh1_noise_intens -= rate * 10
        time.sleep(0.0001)


if __name__ == '__main__':
    s = Server(winhost='mme').boot()
    s.start()

    mm = Mixer(outs=2, chnls=4, time=.025)

    beat = Metro(time=0.5).play()
    t = CosTable([(0, 0), (100, 1), (500, .3), (8191, 0)])
    bit = Beat(time=.125, taps=16, w1=90, w2=50, w3=35, poly=1, onlyonce=True)
    beat_prob = -1
    counter = 0
    trig = TrigEnv(beat, table=t, dur=2, mul=1)
    a = RCOsc(mul=trig, freq=trig * 220, sharp=1)

    trig1 = TrigEnv(bit, table=t, dur=0.1, mul=1)
    noiz = Freeverb(Noise(mul=trig1))

    LPSine = Sine(freq=0.3, mul=500, add=700)
    LPSine1 = Sine(freq=0.4, mul=500, add=700)

    sine_beat = Beat(time=0.25, taps=16, w1=[90, 80], w2=50, w3=35, poly=1).play()
    trmid = TrigXnoiseMidi(sine_beat, dist=12, mrange=(60, 96))
    trhz = Snap(trmid, choice=[0, 2, 3, 5, 7, 8, 10], scale=1)
    tr2 = TrigEnv(sine_beat, table=t, dur=sine_beat['dur'], mul=sine_beat['amp'])
    sup_saw = Freeverb(Delay(SuperSaw(freq=trhz/2, mul=tr2*2, detune=0.5)), size=1, bal=0.5, damp=0.1)

    sine_beat1 = Beat(time=1, taps=16, w1=[90, 80], w2=50, w3=35, poly=1).play()
    trmid1 = TrigXnoiseMidi(sine_beat1, dist=12, mrange=(60, 96))
    trhz1 = Snap(trmid1, choice=[0, 2, 3, 5, 7, 8, 10], scale=1)
    tr21 = TrigEnv(sine_beat1, table=t, dur=sine_beat1['dur'], mul=sine_beat1['amp'])
    sup_saw1 = Freeverb(Delay(SuperSaw(freq=trhz1 / 4, mul=tr21 * 2, detune=0.6)), size=1, bal=0.3, damp=0)

    def trig_func():
        global mesh1_noise_intens
        global counter
        # cam_obj.translation = (random.uniform(-3, 3), random.uniform(0.2, 1.5), random.uniform(-3, 3))
        # cam_obj.rotation = (0, 0, random.uniform(-2, 2))
        counter = (counter + 1) % 10
        mesh1_noise_intens += random.uniform(2, float(counter*4))
        global beat_prob
        beat_prob = min(50, max(beat_prob + random.randint(-5, 5), 0))
        if random.randint(0, 100) < beat_prob:
            bit.play()

    trigfunc = TrigFunc(sine_beat1, trig_func)
    sine_shift = Sine(freq=0.5, mul=5)
    sup_saw = FreqShift(sup_saw, sine_shift)
    sup_saw1 = FreqShift(sup_saw1, sine_shift)

    b = Freeverb(a)
    # mm.addInput(0, b)
    # mm.setAmp(0, 0, 0.1)
    # mm.setAmp(0, 1, 0.1)
    # mm.addInput(1, noiz)
    # mm.setAmp(1, 0, 0.25)
    # mm.setAmp(1, 1, 0.25)
    mm.addInput(2, sup_saw)
    mm.setAmp(2, 0, 0.3)
    mm.setAmp(2, 1, 0.3)
    mm.addInput(3, sup_saw1)
    mm.setAmp(3, 0, 0.4)
    mm.setAmp(3, 1, 0.4)
    mm.out()
    funn()
