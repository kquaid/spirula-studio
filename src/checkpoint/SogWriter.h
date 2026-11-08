#pragma once

// A SplatCloud written as a bundled PlayCanvas SOG v2 (.sog, a zip of lossless
// WebP images and meta.json). Positions, rotations and scales are stored in
// the cloud's own frame, so the result lines up with the splat.ply it came
// from. Spherical-beta lobes are not representable and are dropped.

#include "checkpoint/SplatPly.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace spirula {

struct SogStats {
    int64_t count = 0;
    int sh_bands = 0;
    int palette = 0;
    size_t bytes = 0;
};

// Throws std::runtime_error on an empty or malformed cloud or an unwritable path.
SogStats write_sog(const SplatCloud& cloud, const std::string& path);

}  // namespace spirula
