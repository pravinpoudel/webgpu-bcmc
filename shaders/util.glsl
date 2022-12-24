#ifndef UTIL_GLSL
#define UTIL_GLSL

#define UINT_MAX uint(0xffffffff)
#define FLT_MAX ( 3.402823466e+38f )

struct RayInfo {
    vec3 ray_dir;
    uint block_id;
    float t;
    float t_next;
    // NOTE: std430 layout rules dictate the struct alignment is that of its
    // largest member, which is the vec3 ray dir (whose alignment is same as vec4).
    // This results in the struct size rounding up to 32, since it has to start
    // on 16 byte boundaries.
    // So we have a free 4 byte value to use if needed.
};

struct BlockRange {
    vec2 range;
    float corners[8];
};

struct CoarsedBlockRange{
    vec2 range;
};

struct BlockInfo {
    uint id;
    uint ray_offset;
    uint num_rays;
    uint lod;
};


struct GridIterator {
    ivec3 grid_dims;
    ivec3 grid_step;
    vec3 t_delta;

    ivec3 cell;
    vec3 t_max;
    float t;
};

bool outside_grid(const vec3 p, const vec3 grid_dims) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, grid_dims));
}

bool outside_dual_grid(const vec3 p, const vec3 grid_dims) {
    return any(lessThan(p, vec3(0))) || any(greaterThanEqual(p, grid_dims - vec3(1)));
}

// {
//     grid_dims, grid_step (sign), t_delta (inverse_ray), cell, tmax, t     
// }
GridIterator init_grid_iterator(vec3 ray_org, vec3 ray_dir, float t, ivec3 grid_dims) {
    GridIterator grid_iter;
    grid_iter.grid_dims = grid_dims;
    grid_iter.grid_step = ivec3(sign(ray_dir));

    const vec3 inv_ray_dir = 1.0 / ray_dir;
    grid_iter.t_delta = abs(inv_ray_dir);

	vec3 p = (ray_org + t * ray_dir);
    p = clamp(p, vec3(0), vec3(grid_dims - 1));
    vec3 cell = floor(p);
    const vec3 t_max_neg = (cell - ray_org) * inv_ray_dir;
    const vec3 t_max_pos = (cell + vec3(1) - ray_org) * inv_ray_dir;

    // Pick between positive/negative t_max based on the ray sign
    const bvec3 is_neg_dir = lessThan(ray_dir, vec3(0));
    grid_iter.t_max = mix(t_max_pos, t_max_neg, is_neg_dir);

    grid_iter.cell = ivec3(cell);

    grid_iter.t = t;

    return grid_iter;
}


bool grid_iterator_next_cell(inout GridIterator iter, out vec2 cell_t_range, out ivec3 cell_id) {
    if (outside_grid(iter.cell, iter.grid_dims)) {
        return false;
    }
    // Return the current cell range and ID to the caller
    cell_t_range.x = iter.t;
    cell_t_range.y = min(iter.t_max.x, min(iter.t_max.y, iter.t_max.z));
    cell_id = iter.cell;
    if (cell_t_range.y < cell_t_range.x) {
        return false;
    }

    // Move the iterator to the next cell we'll traverse
    iter.t = cell_t_range.y;
    if (iter.t == iter.t_max.x) {
        iter.cell.x += iter.grid_step.x;
        iter.t_max.x += iter.t_delta.x;
    } else if (iter.t == iter.t_max.y) {
        iter.cell.y += iter.grid_step.y;
        iter.t_max.y += iter.t_delta.y;
    } else {
        iter.cell.z += iter.grid_step.z;
        iter.t_max.z += iter.t_delta.z;
    }
    return true;
}

#endif

