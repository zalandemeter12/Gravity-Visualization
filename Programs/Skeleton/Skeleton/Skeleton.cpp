// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Demeter Zalán
// Neptun : VERF1U
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 200;

struct Camera {
	vec3 wEye, wLookat, wVup;
public:
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
};

struct PerspectiveCamera : Camera {
	float fov, asp, fp, bp;
public:
	PerspectiveCamera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75 * (float)M_PI / 180.0f;
		fp = 0.01; bp = 100;
	}
	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp),	0, 0, 0,
					0,							1 / tan(fov / 2),	0,							0,
					0,							0,					-(fp + bp) / (bp - fp),		-1,
					0,							0,					-2 * fp * bp / (bp - fp),	0);
	}
};

struct OrtographicCamera : Camera {
	float w, h, f, n;
public:
	OrtographicCamera(): w(2.0f), h(2.0f), f(100.0f), n(0.0f) {}
	mat4 P() {
		return mat4(2.0f / w, 0, 0, 0,
			0, 2.0f / h, 0, 0,
			0, 0, (-1.0f) / (f - n), 0,
			0, 0, -((f + n) / (f - n)), 1.0f);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

mat4 qRotMatrix(vec4 q) {
	float qr = q.w, qi = q.x, qj = q.y, qk = q.z;
	return mat4(1-2*(powf(qj,2) + powf(qk,2)),	2*(qi*qj-qk*qr),				2*(qi*qk+qj*qr),				0,
				2*(qi*qj+qk*qr),				1-2*(powf(qi,2) + powf(qk,2)),	2*(qj*qk-qi*qr),				0,
				2*(qi*qk-qj*qr),				2*(qj*qk+qi*qr),				1-2*(powf(qi,2)+ powf(qj,2)),	0,
				0,								0,								0,								0);
}

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
	virtual void Animate(float tstart, float tend, vec4 p1, vec4 p2) {
		float t = tend;
		vec4 q = vec4(cosf(t / 4.0f), sinf(t / 4.0f) * cosf(t) / 2.0f, sinf(t / 4.0f) * sinf(t) / 2.0f, sin(t / 4.0f) * sqrtf(3.0f / 4.0f));
		vec4 qi = vec4(-1.0f * cosf(t / 4.0f), -1.0f * sinf(t / 4.0f) * cosf(t) / 2.0f, -1.0f * sinf(t / 4.0f) * sinf(t) / 2.0f, sin(t / 4.0f) * sqrtf(3.0f / 4.0f));
		wLightPos = p2 + (p1 - p2) * qRotMatrix(q);
	}
};

class SingleColorTexture : public Texture {
public:
	SingleColorTexture(vec4 color) : Texture() {
		std::vector<vec4> image(1);
		image[0] = color;
		create(1, 1, image, GL_NEAREST);
	}
};

struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int   nLights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;
		in  vec3 wView; 
		in  vec3 wLight[8];
		in  vec2 texcoord;
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(length(lights[i].wLightPos),2);
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class PhongShaderSheet : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights; 
		uniform int   nLights;
		uniform vec3  wEye;       

		layout(location = 0) in vec3  vtxPos;       
		layout(location = 1) in vec3  vtxNorm;     
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		 
		out vec3 wView;          
		out vec3 wLight[8];		 
		out vec2 texcoord;
		out float z;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			z = vtxPos.z;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights; 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;  
		in  vec3 wView;         
		in  vec3 wLight[8];    
		in  vec2 texcoord;
		in float z;
		
        out vec4 fragmentColor; 

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
			
			if (z < -0.05) ka = vec3(0.3, 0.3, 0.3) * texColor;
			if (z < -0.1) ka = vec3(0.27, 0.27, 0.27) * texColor; 
			if (z < -0.15) ka = vec3(0.24, 0.24, 0.24) * texColor;
			if (z < -0.2) ka = vec3(0.21, 0.21, 0.21) * texColor;
			if (z < -0.25) ka = vec3(0.18, 0.18, 0.18) * texColor; 
			if (z < -0.3) ka = vec3(0.15, 0.15, 0.15) * texColor;
			if (z < -0.35) ka = vec3(0.12, 0.12, 0.12) * texColor;
			if (z < -0.5) ka = vec3(0.09, 0.09, 0.09) * texColor;
			if (z < -0.45) ka = vec3(0.06, 0.06, 0.06) * texColor; 
			if (z < -0.5) ka = vec3(0.03, 0.03, 0.03) * texColor; 

			if (z < -0.05) kd = vec3(0.3, 0.3, 0.3) * texColor;
			if (z < -0.1) kd = vec3(0.27, 0.27, 0.27) * texColor; 
			if (z < -0.15) kd = vec3(0.24, 0.24, 0.24) * texColor;
			if (z < -0.2) kd = vec3(0.21, 0.21, 0.21) * texColor;
			if (z < -0.25) kd = vec3(0.18, 0.18, 0.18) * texColor; 
			if (z < -0.3) kd = vec3(0.15, 0.15, 0.15) * texColor;
			if (z < -0.35) kd = vec3(0.12, 0.12, 0.12) * texColor;
			if (z < -0.5) kd = vec3(0.09, 0.09, 0.09) * texColor;
			if (z < -0.45) kd = vec3(0.06, 0.06, 0.06) * texColor; 
			if (z < -0.5) kd = vec3(0.03, 0.03, 0.03) * texColor; 

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(length(lights[i].wLightPos),2);
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShaderSheet() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

struct Weight {
	vec2 position;
	float mass;
	Weight(vec2 position, float mass) {
		this->position = position;
		this->mass = mass;
	}
};

class GravitySheet : public ParamSurface {
public:
	std::vector<Weight*> weights;
	GravitySheet() {
		this->weights = std::vector<Weight*>();
		create();
	}
	GravitySheet(std::vector<Weight*> weights) {
		this->weights = weights;
		create(); 
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 - 1;
		V = V * 2 - 1; 
		X = U;
		Y = V;
		Z = 0;
		for (Weight* weight: weights) {
			Dnum2 Xi = Dnum2(0.5*weight->position.x);
			Dnum2 Yi = Dnum2(0.5*weight->position.y);
			Z = Z - Pow(Pow(Pow(X - Xi, 2) + Pow(Y - Yi, 2), 0.5f) + 0.005 * 2, -1) * (weight->mass);
		}
	}
};

struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

struct SheetObject : Object {
	SheetObject(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Object(_shader,_material,_texture,_geometry) {}
	void Animate(float tstart, float tend) {}
	void addWeight(Weight* weight) {
		((GravitySheet*)geometry)->weights.push_back(weight);
	}
	int getWeightNum() {
		return ((GravitySheet*)geometry)->weights.size();
	}
	std::vector<Weight*> getWeights() {
		return ((GravitySheet*)geometry)->weights;
	}
};

struct SphereObject : Object {
	vec3 velocity = vec3(0, 0, 0);
	vec3 intersect = vec3(-0.95,-0.95, 0);
	vec3 normal = vec3(0, 0, 1);
	SphereObject(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Object(_shader, _material, _texture, _geometry) {
		translation = vec3(-0.95, -0.95, 0.05);
	}
	
	VertexData getIntersect(float u, float v, SheetObject* sheetObject) {
		return ((GravitySheet*)sheetObject->geometry)->GenVertexData((u + 1.0f) / 2.0f, (v + 1.0f) / 2.0f);
	}
	
	void Animate(float tstart, float tend, std::vector<Weight*>& weights, PerspectiveCamera* perspectiveCamera, std::vector<SphereObject*>& spheres, SheetObject* sheetObject) {		
		if (length(velocity) > 0.0001) {
			VertexData vtxData = getIntersect(intersect.x, intersect.y, sheetObject);
			normal = normalize(vtxData.normal);
			intersect = vtxData.position;

			float dt = tend - tstart;
			vec3 g = vec3(0.0f, 0.0f, -9.81f);
			vec3 u = dot(g, normal) * normal;
			vec3 a = g - u;

			velocity = velocity + a * dt;
			intersect = intersect + velocity * dt;

			vtxData = getIntersect(intersect.x, intersect.y, sheetObject);
			normal = normalize(vtxData.normal);
			intersect = vtxData.position;
			translation = intersect + normal * 0.05;

			if (this == spheres[0]) {
				perspectiveCamera->wVup = normal;
				vec3 tmp = cross(normal, velocity);
				float theta = 90 * M_PI / 180;
				vec3 rotVel = normal * cosf(theta) + cross(tmp, normal) * sinf(theta) + tmp * dot(tmp, normal) * (1 - cosf(theta));
				perspectiveCamera->wLookat = translation + normalize(rotVel)*0.05;
			}
		}

		if (intersect.x > 1.0f) intersect.x -= 2.0f;
		if (intersect.y > 1.0f) intersect.y -= 2.0f;
		if (intersect.x < -1.0f) intersect.x += 2.0f;
		if (intersect.y < -1.0f) intersect.y += 2.0f;

		for (Weight* weight : weights) {
			if (length(vec3(translation.x, translation.y, 0) - vec3(weight->position.x, weight->position.y, 0)) < 0.035) {
				for (int i = 0; i < spheres.size(); ++i) {
					if (spheres[i] == this) {
						spheres.erase(spheres.begin() + i);
						break;
					}
				}
			}
		}

		
	}
};

float randomFloat(float LO, float HI) {
	//Random float generation from here:
	//https://stackoverflow.com/questions/686353/random-float-number-generation
	return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

//HSV to RGB conversion using the formula from here: 
//https://www.rapidtables.com/convert/color/hsv-to-rgb.html
vec4 HSVtoRGB(float H, float S, float V) {
	float C = S * V;
	float X = C * (1 - fabs(fmod(H / 60.0, 2) - 1));
	float m = V - C;
	if (0 <= H && H < 60) { return vec4((C + m), (X + m), (0 + m), 1); }
	if (60 <= H && H < 120) { return vec4((X + m), (C + m), (0 + m), 1); }
	if (120 <= H && H < 180) { return vec4((0 + m), (C + m), (X + m), 1); }
	if (180 <= H && H < 240) { return vec4((0 + m), (X + m), (C + m), 1); }
	if (240 <= H && H < 300) { return vec4((X + m), (0 + m), (C + m), 1); }
	if (300 <= H && H < 360) { return vec4((C + m), (0 + m), (X + m), 1); }
}

class Scene {
	OrtographicCamera* ortographicCamera;
	PerspectiveCamera* perspectiveCamera;
	std::vector<Light> lights;
	vec4 origo0, origo1;
public:
	bool followSphere = false;
	std::vector<SphereObject*> spheres;
	SheetObject* sheetObject;
	Shader* phongShader;
	Shader* phongShaderSheet;
	Material* sphereMaterial;
	Material* sheetMaterial;
	std::vector<Weight*> weights;
	float weightNum = 0.25;
	void Build() {
		ortographicCamera = new OrtographicCamera();
		perspectiveCamera = new PerspectiveCamera();
		
		phongShader = new PhongShader();
		phongShaderSheet = new PhongShaderSheet();

		sphereMaterial = new Material();
		sphereMaterial->kd = vec3(0.8f, 0.6f, 0.4f);
		sphereMaterial->ks = vec3(0.3f, 0.3f, 0.3f);
		sphereMaterial->ka = vec3(0.2f, 0.2f, 0.2f);
		sphereMaterial->shininess = 30;

		int hue = rand() % 360;
		float saturation = 0.8 + randomFloat(0, 1) * 0.2;
		float value = 0.8 + randomFloat(0, 1) * 0.2;

		SphereObject* sphereObject = new SphereObject(phongShader, sphereMaterial, new SingleColorTexture(HSVtoRGB(hue, saturation, value)), new Sphere());
		sphereObject->scale = vec3(0.05f, 0.05f, 0.05f);
		spheres.push_back(sphereObject);

		sheetMaterial = new Material();
		sheetMaterial->kd = vec3(0.3, 0.3, 0.3);
		sheetMaterial->ks = vec3(0.5, 0.5, 0.5);
		sheetMaterial->ka = vec3(0.3, 0.3, 0.3);
		sheetMaterial->shininess = 10;
		
		sheetObject = new SheetObject(phongShaderSheet, sheetMaterial, new SingleColorTexture(vec4(0.3, 0.2, 1, 1)), new GravitySheet());
		sheetObject->translation = vec3(0, 0, 0);
		sheetObject->scale = vec3(2.f, 2.f, 2.f);	

		lights.resize(2);
		
		origo0 = vec4(-0.2, 0.5, 0.75, 1);
		lights[0].wLightPos = origo0;
		lights[0].La = vec3(1, 1, 1);
		lights[0].Le = vec3(2, 2, 2);

		origo1 = vec4(0.1, -0.3, 0.75, 1);
		lights[1].wLightPos = origo1;
		lights[1].La = vec3(1, 1, 1);
		lights[1].Le = vec3(2, 2, 2);
	}

	void Render() {
		RenderState state;
		if (followSphere && spheres.size()>0) {
			if (length(spheres[0]->velocity) < 0.0001)
				perspectiveCamera->wLookat = spheres[0]->translation + vec3(0.5, 0.5, 0);
			else 
				perspectiveCamera->wLookat = spheres[0]->translation + spheres[0]->velocity;

			perspectiveCamera->wVup = spheres[0]->normal;
			perspectiveCamera->wEye = spheres[0]->translation;

			state.wEye = perspectiveCamera->wEye;
			state.V = perspectiveCamera->V();
			state.P = perspectiveCamera->P();
			state.lights = lights;
			sheetObject->Draw(state);
			for (SphereObject* sphere : spheres)
				if (sphere != spheres[0])
					sphere->Draw(state);
		}
		else {
			ortographicCamera->wEye = vec3(0, 0, 1);
			ortographicCamera->wLookat = vec3(0, 0, 0);
			ortographicCamera->wVup = vec3(0, 1, 0);
			state.wEye = ortographicCamera->wEye;
			state.V = ortographicCamera->V();
			state.P = ortographicCamera->P();
			state.lights = lights;
			for (Object* obj : spheres) obj->Draw(state);
			sheetObject->Draw(state);
		}
	}

	void Animate(float tstart, float tend) {
		for (SphereObject* obj : spheres) obj->Animate(tstart, tend, weights, perspectiveCamera, spheres, sheetObject);
		lights[0].Animate(tstart, tend, origo0, origo1);
		lights[1].Animate(tstart, tend, origo1, origo0);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') scene.followSphere = !scene.followSphere;
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		scene.spheres[scene.spheres.size() - 1]->velocity = vec3(cX + 0.95, cY + 0.95, 0);

		int hue = rand() % 360;
		float saturation = 0.8 + randomFloat(0, 1) * 0.2;
		float value = 0.8 + randomFloat(0, 1) * 0.2;
		
		SphereObject* newSphereObject = new SphereObject(scene.phongShader, scene.sphereMaterial, new SingleColorTexture(HSVtoRGB(hue, saturation, value)), new Sphere());
		newSphereObject->scale = vec3(0.05f, 0.05f, 0.05f);

		scene.spheres.push_back(newSphereObject);
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight; 

		scene.weightNum += 0.25;
		Weight* weight = new Weight(vec2(cX, cY), (scene.weightNum) * 0.01f);
		scene.weights.push_back(weight);

		scene.sheetObject = new SheetObject(scene.phongShaderSheet, scene.sheetMaterial, new SingleColorTexture(vec4(0.3, 0.2, 1, 1)), new GravitySheet(scene.weights));
		scene.sheetObject->translation = vec3(0, 0, 0);
		scene.sheetObject->scale = vec3(2.f, 2.f, 2.f);	
	}
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.03333f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}