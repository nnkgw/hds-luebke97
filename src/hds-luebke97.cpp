// hds-luebke97.cpp (single-file, buildable)
// Unofficial minimal demo aligned with Luebke & Erikson (1997)
// Stateless LOD: each frame pick proxy = smallest ancestor whose projected
// diameter is >= threshold (or leaf if still >=). This guarantees [ / ] works.

#if defined(WIN32)
#pragma warning(disable:4996)
#include <GL/glut.h>
#include <GL/freeglut.h>
#elif defined(__APPLE__) || defined(MACOSX)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/freeglut.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

// ---------------- Mesh & OBJ loader (v/f only, triangulate fan) ----------------
struct Face { int a,b,c; };
struct Mesh {
  std::vector<glm::vec3> V;
  std::vector<Face> F;
  glm::vec3 bbmin{0}, bbmax{0};
};

static inline int parse_index_token(std::string_view tok) noexcept {
  if (auto s = tok.find('/'); s != std::string_view::npos) tok = tok.substr(0, s);
  if (tok.empty()) return 0;
  int value = 0;
  auto [p, ec] = std::from_chars(tok.data(), tok.data()+tok.size(), value, 10);
  if (ec != std::errc()) return 0;
  return value;
}
static inline int resolve_index(int idx, int n) {
  if (idx > 0) return idx - 1;
  if (idx < 0) return n + idx;
  return -1;
}
static bool load_obj(const std::string& path, Mesh& M) {
  std::ifstream ifs(path);
  if (!ifs) { std::cerr<<"open failed: "<<path<<"\n"; return false; }
  std::vector<glm::vec3> V; std::vector<Face> F;
  std::string line, tok;
  while (std::getline(ifs, line)) {
    if (line.size()<2) continue;
    if (line[0]=='v' && std::isspace((unsigned char)line[1])) {
      std::istringstream iss(line.substr(2));
      float x,y,z; iss>>x>>y>>z; V.emplace_back(x,y,z);
    } else if (line[0]=='f' && std::isspace((unsigned char)line[1])) {
      std::istringstream iss(line.substr(2));
      std::vector<int> idxs;
      while (iss>>tok) {
        int id = resolve_index(parse_index_token(tok), (int)V.size());
        if (id<0 || id>=(int)V.size()) { idxs.clear(); break; }
        idxs.push_back(id);
      }
      if (idxs.size()>=3) for (size_t i=1;i+1<idxs.size();++i)
        F.push_back({idxs[0], (int)idxs[i], (int)idxs[i+1]});
    }
  }
  if (V.empty() || F.empty()) { std::cerr<<"empty mesh\n"; return false; }
  M.V.swap(V); M.F.swap(F);
  glm::vec3 mn( std::numeric_limits<float>::max());
  glm::vec3 mx(-std::numeric_limits<float>::max());
  for (auto& p: M.V){ mn=glm::min(mn,p); mx=glm::max(mx,p); }
  M.bbmin=mn; M.bbmax=mx;
  return true;
}

// ---------------- HDS-style tight octree ----------------
struct Node {
  glm::vec3 repvert{0}; // representative vertex (avg of contained verts or of children reps)
  glm::vec3 center{0};  // for bounding sphere
  float radius{0};

  glm::vec3 bmin{0}, bmax{0};
  Node* parent{nullptr};
  Node* children[8]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
  bool leaf{true};
  std::vector<int> verts; // indices (if leaf)
};

struct Tri { Node* corners[3]{nullptr,nullptr,nullptr}; Node* proxies[3]{nullptr,nullptr,nullptr}; Tri* prev=nullptr; Tri* next=nullptr; bool in_active=false; };

struct HDS {
  Mesh* mesh{nullptr};
  Node* root{nullptr};
  std::vector<Node*> v2leaf;
  std::vector<std::unique_ptr<Tri>> allTris;

  // view / control
  float fovy_rad{45.0f*(float)M_PI/180.0f};
  int viewportW{1280}, viewportH{720};
  float threshold_px{12.0f};
  bool wire{false};
  bool freezeLOD{false};

  // active triangle list head
  Tri* activeHead{nullptr};

  void clear() {
    std::function<void(Node*)> rec=[&](Node* n){ if(!n) return; for(int i=0;i<8;++i) rec(n->children[i]); delete n; };
    rec(root); root=nullptr;
    v2leaf.clear(); allTris.clear(); activeHead=nullptr; mesh=nullptr;
  }

  void buildVertexTree(Mesh& M, int maxDepth=24, int leafSize=1) {
    clear(); mesh=&M;
    root = new Node();
    root->bmin=M.bbmin; root->bmax=M.bbmax;
    glm::vec3 diag = root->bmax - root->bmin;
    float m = std::max({std::abs(diag.x), std::abs(diag.y), std::abs(diag.z)});
    glm::vec3 c = 0.5f*(root->bmin + root->bmax);
    root->bmin = c - 0.5f*glm::vec3(m);
    root->bmax = c + 0.5f*glm::vec3(m);
    root->center = 0.5f*(root->bmin + root->bmax);
    root->radius = 0.5f*glm::length(root->bmax - root->bmin);

    std::vector<int> ids(M.V.size()); for(size_t i=0;i<ids.size();++i) ids[i]=(int)i;
    buildRecursive(root, ids, maxDepth, leafSize);
    computeRepverts(root);

    v2leaf.assign(M.V.size(), nullptr);
    mapLeaves(root);

    for (const Face& f : M.F) {
      auto t = std::make_unique<Tri>();
      t->corners[0]=v2leaf[f.a]; t->corners[1]=v2leaf[f.b]; t->corners[2]=v2leaf[f.c];
      allTris.emplace_back(std::move(t));
    }
  }

  // ---------- per-frame LOD (stateless) ----------
  void updateLOD(const glm::mat4& V) {
    if (!root) return;
    if (freezeLOD) return; // keep current active list
    rebuildActiveListStateless(V);
  }

  // draw
  void renderActiveList(bool wireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, wireframe?GL_LINE:GL_FILL);
    glBegin(GL_TRIANGLES);
    for (Tri* t = activeHead; t; t=t->next) {
      const glm::vec3& A = t->proxies[0]->repvert;
      const glm::vec3& B = t->proxies[1]->repvert;
      const glm::vec3& C = t->proxies[2]->repvert;
      glm::vec3 n = glm::normalize(glm::cross(B-A, C-A));
      glNormal3f(n.x,n.y,n.z);
      glVertex3f(A.x,A.y,A.z);
      glVertex3f(B.x,B.y,B.z);
      glVertex3f(C.x,C.y,C.z);
    }
    glEnd();
  }

  // ---------- internals ----------
  float nodeSizePx(const Node* N, const glm::mat4& V) const {
    glm::vec3 c = glm::vec3(V * glm::vec4(N->center,1));
    float z = std::abs(c.z); if (z < 1e-6f) z = 1e-6f;
    float pixelsPerWorld = (float)viewportH * 0.5f / std::tan(fovy_rad*0.5f);
    float r_px = (N->radius / z) * pixelsPerWorld;
    return 2.0f * r_px;
  }

  Node* chooseProxyByThreshold(Node* leaf, const glm::mat4& V, float thr) const {
    Node* a = leaf;
    // climb up while current cluster is too small on screen
    while (a->parent && nodeSizePx(a, V) < thr) a = a->parent;
    return a;
  }

  void rebuildActiveListStateless(const glm::mat4& V) {
    activeHead=nullptr;
    for (auto& up : allTris) {
      Tri* T = up.get();
      T->prev=T->next=nullptr; T->in_active=false;
      for (int k=0;k<3;++k) T->proxies[k] = chooseProxyByThreshold(T->corners[k], V, threshold_px);
      // drop degenerate clustered tris
      if (T->proxies[0]==T->proxies[1] || T->proxies[1]==T->proxies[2] || T->proxies[2]==T->proxies[0]) continue;
      // push-front
      T->next = activeHead; if (activeHead) activeHead->prev=T; activeHead=T; T->in_active=true;
    }
  }

  void buildRecursive(Node* n, const std::vector<int>& ids, int maxDepth, int leafSize) {
    // recompute tight cubic bounds for stability
    glm::vec3 mn( std::numeric_limits<float>::max());
    glm::vec3 mx(-std::numeric_limits<float>::max());
    for (int vi: ids){ mn=glm::min(mn, mesh->V[vi]); mx=glm::max(mx, mesh->V[vi]); }
    glm::vec3 c = 0.5f*(mn+mx); glm::vec3 e = mx - mn;
    float m = std::max({std::abs(e.x), std::abs(e.y), std::abs(e.z)});
    n->bmin = c - 0.5f*glm::vec3(m);
    n->bmax = c + 0.5f*glm::vec3(m);
    n->center = 0.5f*(n->bmin + n->bmax);
    n->radius = 0.5f*glm::length(n->bmax - n->bmin);

    if ((int)ids.size() <= leafSize || maxDepth<=0) { n->leaf=true; n->verts=ids; return; }
    n->leaf=false;

    glm::vec3 mid = 0.5f*(n->bmin + n->bmax);
    std::vector<int> bucket[8];
    for (int vi: ids) {
      const glm::vec3& p = mesh->V[vi];
      int code = (p.x>mid.x?1:0) | (p.y>mid.y?2:0) | (p.z>mid.z?4:0);
      bucket[code].push_back(vi);
    }
    for (int i=0;i<8;++i) if (!bucket[i].empty()) {
      Node* ch = new Node(); ch->parent=n;
      // child bounds
      glm::vec3 mn2=n->bmin, mx2=n->bmax;
      if (i&1) mn2.x=mid.x; else mx2.x=mid.x;
      if (i&2) mn2.y=mid.y; else mx2.y=mid.y;
      if (i&4) mn2.z=mid.z; else mx2.z=mid.z;
      ch->bmin=mn2; ch->bmax=mx2;
      n->children[i]=ch;
      buildRecursive(ch, bucket[i], maxDepth-1, leafSize);
    }
  }

  void computeRepverts(Node* n) {
    if (!n) return;
    if (n->leaf) {
      if (!n->verts.empty()) {
        glm::vec3 s(0); for (int vi: n->verts) s += mesh->V[vi];
        n->repvert = s / (float)n->verts.size();
      } else n->repvert = n->center;
    } else {
      for (int i=0;i<8;++i) if (n->children[i]) computeRepverts(n->children[i]);
      glm::vec3 s(0); int c=0;
      for (int i=0;i<8;++i) if (n->children[i]) { s += n->children[i]->repvert; ++c; }
      n->repvert = (c>0)?(s/(float)c):n->center;
    }
  }

  void mapLeaves(Node* n) {
    if (!n) return;
    if (n->leaf) {
      for (int vi: n->verts) v2leaf[vi]=n;
    } else {
      for (int i=0;i<8;++i) if (n->children[i]) mapLeaves(n->children[i]);
    }
  }
};

// ---------------- App / camera / UI ----------------
struct App {
  Mesh mesh;
  HDS  hds;
  float yaw=0.0f, pitch=0.0f, dist=3.0f;
  glm::vec3 center{0,0,0};
  int winW=1280, winH=720;
  bool dragging=false; int lastx=0,lasty=0;
} G;

static glm::mat4 viewMatrix() {
  glm::vec3 eye = G.center + glm::vec3(
    G.dist * std::cos(G.pitch)*std::sin(G.yaw),
    G.dist * std::sin(G.pitch),
    G.dist * std::cos(G.pitch)*std::cos(G.yaw));
  return glm::lookAt(eye, G.center, glm::vec3(0,1,0));
}
static glm::mat4 projMatrix() {
  float aspect = (float)G.winW/(float)G.winH;
  return glm::perspective(G.hds.fovy_rad, aspect, 0.01f, 1000.0f);
}

static void drawHUD() {
  glDisable(GL_LIGHTING);
  glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, G.winW, 0, G.winH, -1, 1);
  glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
  glColor3f(1,1,1);
  auto text=[&](int x,int y,const char*s){ glRasterPos2i(x,y); while(*s) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s++); };
  char buf[256];
  std::snprintf(buf,sizeof(buf),
    "F(orig): %zu   threshold: %.2f px   %s   (SPACE: freeze, w: wire, [ ]: thr)",
    G.mesh.F.size(), G.hds.threshold_px, G.hds.freezeLOD?"[frozen]":"");
  text(10, G.winH-20, buf);
}

static void display() {
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.07f,0.08f,0.10f,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  glm::mat4 V = viewMatrix();
  glm::mat4 P = projMatrix();

  glMatrixMode(GL_PROJECTION); glLoadMatrixf(&P[0][0]);
  glMatrixMode(GL_MODELVIEW);  glLoadMatrixf(&V[0][0]);

  // update stateless LOD
  G.hds.viewportW = G.winW; G.hds.viewportH = G.winH;
  G.hds.updateLOD(V);

  // simple lighting
  glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_NORMALIZE);
  float pos[4]={3,4,5,1}; glLightfv(GL_LIGHT0, GL_POSITION, pos);
  float dif[4]={0.9f,0.9f,0.9f,1}; glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
  glColor3f(0.7f,0.85f,1.0f);

  G.hds.renderActiveList(G.hds.wire);
  drawHUD();
  glutSwapBuffers();
}

static void reshape(int w,int h){ G.winW=w; G.winH=(h>1?h:1); glViewport(0,0,G.winW,G.winH); }
static void idle(){ glutPostRedisplay(); }
static void mouse(int button,int state,int x,int y){
  if (button==GLUT_LEFT_BUTTON){ G.dragging=(state==GLUT_DOWN); G.lastx=x; G.lasty=y; }
  if (button==3) G.dist *= 0.9f;
  if (button==4) G.dist *= 1.111f;
}
static void motion(int x,int y){
  if(!G.dragging) return;
  float dx=(float)(x-G.lastx), dy=(float)(y-G.lasty);
  G.lastx=x; G.lasty=y;
  G.yaw += dx*0.005f; G.pitch += dy*0.005f;
  if (G.pitch<-1.5f) G.pitch=-1.5f;
  if (G.pitch> 1.5f) G.pitch= 1.5f;
}
static void keyboard(unsigned char k,int,int){
  if (k==27 || k=='q') std::exit(0);
  if (k=='w') G.hds.wire = !G.hds.wire;
  if (k==' ') G.hds.freezeLOD = !G.hds.freezeLOD;
  if (k=='[') G.hds.threshold_px = std::max(0.01f, G.hds.threshold_px-1.0f);
  if (k==']') G.hds.threshold_px += 1.0f;
  if (k=='r') {
    G.yaw=0; G.pitch=0;
    G.dist = glm::length(G.mesh.bbmax - G.mesh.bbmin) * 1.2f;
    G.hds.threshold_px = 12.0f; G.hds.freezeLOD=false;
  }
}

int main(int argc, char** argv){
  if (argc<2) { std::cerr<<"Usage: "<<argv[0]<<" model.obj\n"; return 1; }
  if (!load_obj(argv[1], G.mesh)) return 1;

  G.hds.buildVertexTree(G.mesh, /*maxDepth*/24, /*leafSize*/1);
  G.hds.threshold_px = 12.0f;

  G.center = 0.5f*(G.mesh.bbmin + G.mesh.bbmax);
  G.dist   = glm::length(G.mesh.bbmax - G.mesh.bbmin) * 1.2f;

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(G.winW, G.winH);
  glutCreateWindow("HDS (stateless LOD) - minimal");

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(idle);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);

  glutMainLoop();
  return 0;
}
