// hds_luebke97.cpp
// Unofficial minimal implementation aligned with
// "View-Dependent Simplification of Arbitrary Polygonal Environments" (Luebke & Erikson, 1997)
// - Vertex tree (tight octree), Tri/Node structures & adjustTree/collapseNode/expandNode per paper
// - Active Triangle List renderer (fixed-function OpenGL for simplicity)
// Note: silhouette test / triangle budget / visibility containers are omitted but hookable.
#if defined(WIN32)
#pragma warning(disable:4996)
#include <GL/glut.h>
#include <GL/freeglut.h>
#ifdef NDEBUG
//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#endif // NDEBUG

#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#elif defined(__APPLE__) || defined(MACOSX)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else // MACOSX
#include <GL/glut.h>
#include <GL/freeglut.h>
#endif // unix

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ----------------------------- Basic mesh -----------------------------
struct Face { int a, b, c; };
struct Mesh {
  std::vector<glm::vec3> V;
  std::vector<Face> F;
  glm::vec3 bbmin{0}, bbmax{0};
};

// OBJ loader helpers (v/f, triangulate fan, negative indices)
static inline int parse_index_token(std::string_view tok) noexcept {
  if (auto s = tok.find('/'); s != std::string_view::npos) tok = tok.substr(0, s);
  if (tok.empty()) return 0;
  int value = 0;
  const char* b = tok.data(); const char* e = b + tok.size();
  auto [p, ec] = std::from_chars(b, e, value, 10);
  if (ec != std::errc()) return 0;
  return value;
}
static inline int resolve_index(int idx, int n) {
  if (idx > 0) return idx - 1;
  if (idx < 0) return n + idx; // OBJ negative index = relative to end
  return -1;
}
static bool load_obj(const std::string& path, Mesh& M) {
  std::ifstream ifs(path);
  if (!ifs) { std::cerr << "open failed: " << path << "\n"; return false; }
  std::vector<glm::vec3> V; std::vector<Face> F;

  std::string line, tok;
  while (std::getline(ifs, line)) {
    if (line.size() < 2) continue;
    if (line[0]=='v' && std::isspace(static_cast<unsigned char>(line[1]))) {
      std::istringstream iss(line.substr(2));
      float x,y,z; iss>>x>>y>>z; V.emplace_back(x,y,z);
    } else if (line[0]=='f' && std::isspace(static_cast<unsigned char>(line[1]))) {
      std::istringstream iss(line.substr(2));
      std::vector<int> idxs;
      while (iss >> tok) {
        int id = resolve_index(parse_index_token(tok), static_cast<int>(V.size()));
        if (id<0 || id>=static_cast<int>(V.size())) { idxs.clear(); break; }
        idxs.push_back(id);
      }
      if (idxs.size() >= 3) {
        for (size_t i=1;i+1<idxs.size();++i) F.push_back({idxs[0], static_cast<int>(idxs[i]), static_cast<int>(idxs[i+1])});
      }
    }
  }
  if (V.empty() || F.empty()) { std::cerr << "empty mesh\n"; return false; }
  M.V.swap(V); M.F.swap(F);

  glm::vec3 mn( std::numeric_limits<float>::max());
  glm::vec3 mx(-std::numeric_limits<float>::max());
  for (const glm::vec3& p : M.V) { mn = glm::min(mn,p); mx = glm::max(mx,p); }
  M.bbmin = mn; M.bbmax = mx;
  return true;
}

// ----------------------------- HDS structures (paper-aligned) -----------------------------
struct Node; // forward

// Åò3.1 Tri ? corners/proxies/prev/next
struct Tri {
  Node* corners[3]{nullptr,nullptr,nullptr}; // leaves (original mesh corners mapped to leaf nodes)
  Node* proxies[3]{nullptr,nullptr,nullptr}; // current first active ancestors
  Tri* prev{nullptr};
  Tri* next{nullptr};
  bool in_active{false};
};

// Åò3.2 Node ? labels and lists (tris/subtris)
enum class NodeStatus : unsigned char { Active, Boundary, Inactive };

struct Node {
  // identification (optional)
  std::uint64_t idbits{0};
  unsigned char depth{0};

  NodeStatus label{NodeStatus::Inactive};

  glm::vec3 repvert{0};   // representative vertex for this node
  glm::vec3 center{0};    // bounding sphere center
  float      radius{0};   // bounding sphere radius

  // lists per paper: tris (exactly one corner in the node), subtris (2?3 corners in node but no child gets 2+)
  std::vector<Tri*> tris;
  std::vector<Tri*> subtris;

  Node* parent{nullptr};
  unsigned char numchildren{0};
  Node* children[8]{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};

  // build-time helpers
  bool leaf{true};
  glm::vec3 bmin{0}, bmax{0};
  std::vector<int> verts; // original vertex indices if leaf
};

// ----------------------------- HDS system -----------------------------
struct HDS {
  Mesh* mesh{nullptr};
  Node* root{nullptr};

  // view / threshold
  float fovy_rad{45.0f * static_cast<float>(M_PI/180.0)};
  int viewportW{1280}, viewportH{720};
  float threshold_px{12.0f};
  bool freezeLOD{false};
  bool wire{false};

  // Active Triangle List head
  Tri* activeHead{nullptr};

  // lifetime-owned triangles
  std::vector<std::unique_ptr<Tri>> allTris;

  // map original vertex -> leaf node
  std::vector<Node*> v2leaf;

  // ---------- build ----------
  void clear() {
    std::function<void(Node*)> rec = [&](Node* n){
      if (!n) return;
      for (int i=0;i<8;++i) rec(n->children[i]);
      delete n;
    };
    rec(root); root=nullptr;
    allTris.clear(); activeHead=nullptr; v2leaf.clear(); mesh=nullptr;
  }

  void buildVertexTree(Mesh& M, int maxDepth=8, int leafSize=128) {
    clear();
    mesh = &M;

    root = new Node();
    root->depth = 0; root->idbits = 0;
    root->bmin = M.bbmin; root->bmax = M.bbmax;
    root->center = 0.5f*(root->bmin + root->bmax);
    root->radius = 0.5f*glm::length(root->bmax - root->bmin);

    std::vector<int> ids(M.V.size());
    for (size_t i=0;i<ids.size();++i) ids[i] = static_cast<int>(i);
    buildRecursive(root, ids, maxDepth, leafSize);
    computeRepverts(root);

    // leaves map
    v2leaf.assign(M.V.size(), nullptr);
    collectLeafMap(root);

    // create Tri list
    for (const Face& f : M.F) {
      std::unique_ptr<Tri> tri = std::make_unique<Tri>();
      tri->corners[0] = v2leaf[f.a];
      tri->corners[1] = v2leaf[f.b];
      tri->corners[2] = v2leaf[f.c];
      allTris.emplace_back(std::move(tri));
    }

    // classify triangles to node lists
    for (const std::unique_ptr<Tri>& up : allTris) classifyTriangleIntoNodeLists(up.get());

    // initial labels: root is Boundary; children inactive
    root->label = NodeStatus::Boundary;
    for (int i=0;i<8;++i) if (root->children[i]) root->children[i]->label = NodeStatus::Inactive;

    // init proxies; active list initially empty
    for (const std::unique_ptr<Tri>& up : allTris) refreshTriProxies(*up);
    activeHead = nullptr;
  }

  // ---------- runtime LOD ----------
  void updateLOD(const glm::mat4& V) {
    if (!root || freezeLOD) return;
    adjustTree(root, V);
  }

  void adjustTree(Node* N, const glm::mat4& V) {
    if (!N) return;
    float size = nodeSize(N, V);
    float thr  = threshold_px; // hook: silhouette-aware threshold can be inserted here

    if (size >= thr) {
      if (N->label == NodeStatus::Active) {
        for (int i=0;i<8;++i) if (N->children[i]) adjustTree(N->children[i], V);
      } else { // Boundary
        expandNode(N);
      }
    } else { // size < thr
      if (N->label == NodeStatus::Active) {
        collapseNode(N);
      }
    }
  }

  // ---------- render ----------
  void renderActiveList(bool wireframe) {
    glPolygonMode(GL_FRONT_AND_BACK, wireframe?GL_LINE:GL_FILL);
    glBegin(GL_TRIANGLES);
    for (Tri* t = activeHead; t; t = t->next) {
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

  // ---------- core ops: collapse / expand ----------
  void collapseNode(Node* N) {
    // deactivate children first
    for (int i=0;i<8;++i) if (Node* C=N->children[i]) {
      if (C->label == NodeStatus::Active) collapseNode(C);
      C->label = NodeStatus::Inactive;
    }
    N->label = NodeStatus::Boundary;

    // update proxies for triangles with one corner in N
    for (Tri* T : N->tris) refreshTriProxies(*T);

    // remove subtris (2?3 corners in N) from Active List
    for (Tri* T : N->subtris) removeTri(T);
  }

  void expandNode(Node* N) {
    // parent becomes Active; children become Boundary
    N->label = NodeStatus::Active;
    for (int i=0;i<8;++i) if (Node* C=N->children[i]) C->label = NodeStatus::Boundary;

    // update proxies for triangles with one corner in N
    for (Tri* T : N->tris) refreshTriProxies(*T);

    // refresh proxies for subtris too, then add (skip degenerates inside addTri)
    for (Tri* T : N->subtris) refreshTriProxies(*T);
    for (Tri* T : N->subtris) addTri(T);
  }

  // ---------- Active list ops ----------
  void addTri(Tri* T) {
    if (T->in_active) return;
    if (T->proxies[0]==T->proxies[1] || T->proxies[1]==T->proxies[2] || T->proxies[2]==T->proxies[0]) return;
    T->next = activeHead; T->prev = nullptr;
    if (activeHead) activeHead->prev = T;
    activeHead = T;
    T->in_active = true;
  }
  void removeTri(Tri* T) {
    if (!T->in_active) return;
    if (T->prev) T->prev->next = T->next; else activeHead = T->next;
    if (T->next) T->next->prev = T->prev;
    T->prev = T->next = nullptr;
    T->in_active = false;
  }

  // ---------- helpers ----------
  Node* firstActiveAncestor(Node* n) const {
    Node* p = n;
    while (p && !(p->label==NodeStatus::Active || p->label==NodeStatus::Boundary)) p = p->parent;
    return p ? p : root;
  }
  void refreshTriProxies(Tri& T) {
    for (int k=0;k<3;++k) T.proxies[k] = firstActiveAncestor(T.corners[k]);
  }

  // Åò4.1 projected node size (bounding sphere diameter in pixels)
  float nodeSize(const Node* N, const glm::mat4& V) const {
    glm::vec3 c = glm::vec3(V * glm::vec4(N->center,1.0f));
    float z = std::abs(c.z);
    if (z < 1e-4f) return 0.0f;
    float pixelsPerWorld = static_cast<float>(viewportH) * 0.5f / std::tan(fovy_rad*0.5f);
    float r_px = (N->radius / z) * pixelsPerWorld;
    return 2.0f * r_px;
  }

private:
  // build a tight octree
  void buildRecursive(Node* n, const std::vector<int>& ids, int maxDepth, int leafSize) {
    n->leaf = (n->depth >= maxDepth) || (static_cast<int>(ids.size()) <= leafSize);

    glm::vec3 mn( std::numeric_limits<float>::max());
    glm::vec3 mx(-std::numeric_limits<float>::max());
    for (int vi : ids) { mn = glm::min(mn, mesh->V[vi]); mx = glm::max(mx, mesh->V[vi]); }
    glm::vec3 c = 0.5f*(mn+mx);
    glm::vec3 e = mx - mn;
    float m = std::max({e.x,e.y,e.z});
    n->bmin = c - 0.5f*glm::vec3(m);
    n->bmax = c + 0.5f*glm::vec3(m);
    n->center = 0.5f*(n->bmin + n->bmax);
    n->radius = 0.5f*glm::length(n->bmax - n->bmin);

    if (n->leaf) {
      n->verts = ids;
      return;
    }

    glm::vec3 mid = 0.5f*(n->bmin + n->bmax);
    std::vector<int> bucket[8];
    for (int vi : ids) {
      const glm::vec3& p = mesh->V[vi];
      int code = (p.x>mid.x?1:0) | (p.y>mid.y?2:0) | (p.z>mid.z?4:0);
      bucket[code].push_back(vi);
    }
    for (int cidx=0;cidx<8;++cidx) if (!bucket[cidx].empty()) {
      Node* ch = new Node();
      ch->parent = n; ch->depth = static_cast<unsigned char>(n->depth + 1);
      ch->idbits = n->idbits | (static_cast<std::uint64_t>(cidx) << (3*ch->depth - 3));

      glm::vec3 mn2 = n->bmin, mx2 = n->bmax;
      if (cidx&1) mn2.x = mid.x; else mx2.x = mid.x;
      if (cidx&2) mn2.y = mid.y; else mx2.y = mid.y;
      if (cidx&4) mn2.z = mid.z; else mx2.z = mid.z;
      ch->bmin = mn2; ch->bmax = mx2;

      n->children[cidx] = ch; n->numchildren++;
      buildRecursive(ch, bucket[cidx], maxDepth, leafSize);
    }
  }

  void computeRepverts(Node* n) {
    if (!n) return;
    if (n->leaf) {
      if (!n->verts.empty()) {
        glm::vec3 s(0); for (int vi : n->verts) s += mesh->V[vi];
        n->repvert = s / static_cast<float>(n->verts.size());
      } else {
        n->repvert = n->center;
      }
    } else {
      for (int i=0;i<8;++i) if (n->children[i]) computeRepverts(n->children[i]);
      glm::vec3 s(0); int cnt=0;
      for (int i=0;i<8;++i) if (n->children[i]) { s += n->children[i]->repvert; ++cnt; }
      n->repvert = (cnt>0) ? (s / static_cast<float>(cnt)) : n->center;
    }
  }

  void collectLeafMap(Node* n) {
    if (!n) return;
    if (n->leaf) {
      for (int vi : n->verts) v2leaf[vi] = n;
    } else {
      for (int i=0;i<8;++i) if (n->children[i]) collectLeafMap(n->children[i]);
    }
  }

  static bool isDescendant(Node* node, Node* ancestor) {
    for (Node* p=node; p; p=p->parent) if (p==ancestor) return true;
    return false;
  }
  static int childIndexOnPath(Node* node, Node* ancestor) {
    Node* p = node;
    while (p && p->parent != ancestor) p = p->parent;
    if (!p || !ancestor) return -1;
    for (int i=0;i<8;++i) if (ancestor->children[i]==p) return i;
    return -1;
  }

  void classifyTriangleIntoNodeLists(Tri* T) {
    // Count appearances of the three corner-leaves in each ancestor up to root
    std::unordered_map<Node*, int> count;
    auto addAnc=[&](Node* x){
      for (Node* p=x; p; p=p->parent) count[p]++;
    };
    addAnc(T->corners[0]); addAnc(T->corners[1]); addAnc(T->corners[2]);

    for (const auto& kv : count) {
      Node* N = kv.first; int cnt = kv.second;
      if (cnt == 1) {
        N->tris.push_back(T);
      } else if (cnt >= 2) {
        bool ok = true;
        if (!N->leaf) {
          int occ[8]={0};
          for (int k=0;k<3;++k) if (isDescendant(T->corners[k], N)) {
            int ci = childIndexOnPath(T->corners[k], N);
            if (ci>=0) { occ[ci]++; if (occ[ci]>1) { ok=false; break; } }
          }
        }
        if (ok) N->subtris.push_back(T);
      }
    }
  }
};

// ----------------------------- App / camera -----------------------------
struct App {
  Mesh mesh;
  HDS  hds;

  // camera (orbit)
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
  float aspect = static_cast<float>(G.winW)/static_cast<float>(G.winH);
  return glm::perspective(G.hds.fovy_rad, aspect, 0.01f, 1000.0f);
}

// ----------------------------- GLUT callbacks -----------------------------
static void drawHUD() {
  glDisable(GL_LIGHTING);
  glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, G.winW, 0, G.winH, -1, 1);
  glMatrixMode(GL_MODELVIEW);  glLoadIdentity();
  glColor3f(1,1,1);
  auto text=[&](int x,int y,const char* s){ glRasterPos2i(x,y); while(*s) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *s++); };
  char buf[256];
  std::snprintf(buf,sizeof(buf),
    "F(orig): %zu   threshold: %.1f px   %s    (SPACE: freeze, w: wire, [ ]: thr)",
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

  // update HDS
  G.hds.viewportW = G.winW; G.hds.viewportH = G.winH;
  G.hds.updateLOD(V);

  // lighting
  glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glDisable(GL_COLOR_MATERIAL);
  float pos[4] = { 3,4,5,1 }; glLightfv(GL_LIGHT0, GL_POSITION, pos);
  float dif[4] = { 0.9f,0.9f,0.9f,1 }; glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);

  glColor3f(0.6f,0.8f,1.0f);
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
  float dx = static_cast<float>(x - G.lastx);
  float dy = static_cast<float>(y - G.lasty);
  G.lastx=x; G.lasty=y;
  G.yaw   += dx * 0.005f;
  G.pitch += dy * 0.005f;
  if (G.pitch < -1.5f) G.pitch = -1.5f;
  if (G.pitch >  1.5f) G.pitch =  1.5f;
}
static void keyboard(unsigned char k,int,int){
  if (k==27 || k=='q') std::exit(0);
  if (k=='w') G.hds.wire = !G.hds.wire;
  if (k==' ') G.hds.freezeLOD = !G.hds.freezeLOD;
  if (k=='[') G.hds.threshold_px = std::max(2.0f, G.hds.threshold_px-1.0f);
  if (k==']') G.hds.threshold_px += 1.0f;
  if (k=='r') {
    G.yaw=0; G.pitch=0; G.dist = glm::length(G.mesh.bbmax - G.mesh.bbmin) * 1.5f;
    G.hds.threshold_px = 12.0f; G.hds.freezeLOD = false;
  }
}

// ----------------------------- main -----------------------------
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " model.obj\n";
    return 1;
  }
  if (!load_obj(argv[1], G.mesh)) return 1;

  G.center = 0.5f*(G.mesh.bbmin + G.mesh.bbmax);
  G.dist   = glm::length(G.mesh.bbmax - G.mesh.bbmin) * 1.2f;

  G.hds.buildVertexTree(G.mesh, /*maxDepth*/8, /*leafSize*/128);

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(G.winW, G.winH);
  glutCreateWindow("HDS (View-Dependent Simplification) ? paper-aligned minimal impl");

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutIdleFunc(idle);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);

  glEnable(GL_NORMALIZE);
  glutMainLoop();
  return 0;
}
