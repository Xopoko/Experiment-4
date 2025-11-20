#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

extern "C" {
struct Vec3 {int x,y,z;};

static const int DIRS[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};

static const int ROT_PERMS[6][3] = {
    {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
};

static const int SIGNS[8][3] = {
    {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},
    {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}
};

static int parity(const int perm[3]) {
    int inv=0;
    for(int i=0;i<3;i++) for(int j=i+1;j<3;j++) if(perm[i]>perm[j]) inv++;
    return (inv%2)==0 ? 1: -1;
}

static std::array<int,3> apply_rot(const std::array<int,3>& c, const int perm[3], const int s[3]) {
    return {s[0]*c[perm[0]], s[1]*c[perm[1]], s[2]*c[perm[2]]};
}

static void normalize(std::vector<std::array<int,3>>& pts) {
    int minx=pts[0][0], miny=pts[0][1], minz=pts[0][2];
    for(auto &p: pts){
        if(p[0]<minx) minx=p[0];
        if(p[1]<miny) miny=p[1];
        if(p[2]<minz) minz=p[2];
    }
    for(auto &p: pts){ p[0]-=minx; p[1]-=miny; p[2]-=minz; }
    std::sort(pts.begin(), pts.end(), [](const auto&a, const auto&b){return a<b;});
}

// coords: length 3*n_cells, returns minimal canonical ordering into out (same length)
void canonicalize_shape(const int32_t* coords, int n_cells, int32_t* out) {
    std::vector<std::array<int,3>> cells; cells.reserve(n_cells);
    for(int i=0;i<n_cells;i++){
        cells.push_back({coords[3*i], coords[3*i+1], coords[3*i+2]});
    }
    std::vector<std::array<int,3>> best;
    bool init=false;
    for(const auto& perm: ROT_PERMS){
        int perm_arr[3] = {perm[0],perm[1],perm[2]};
        int perm_par = parity(perm_arr);
        for(const auto& sgn: SIGNS){
            if(perm_par*sgn[0]*sgn[1]*sgn[2]!=1) continue;
            std::vector<std::array<int,3>> rot; rot.reserve(n_cells);
            for(const auto& c: cells){
                rot.push_back(apply_rot(c, perm_arr, sgn));
            }
            normalize(rot);
            if(!init || rot<best){
                best.swap(rot); init=true;
            }
        }
    }
    for(int i=0;i<n_cells;i++){
        out[3*i]=best[i][0];
        out[3*i+1]=best[i][1];
        out[3*i+2]=best[i][2];
    }
}
}
