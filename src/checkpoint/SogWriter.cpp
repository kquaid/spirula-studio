#include "checkpoint/SogWriter.h"

#include "core/AtomicFile.h"
#include "core/WebpWriter.h"
#include "external/miniz.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <locale>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spirula {
namespace {

namespace fs = std::filesystem;

// Palette size cap: the final assignment is N * palette * dim multiply-adds on the host.
constexpr int kMaxPalette = 8192;
constexpr float kInvSqrt2 = 0.70710678118654752f;

using Bytes = std::vector<uint8_t>;
using Entry = std::pair<std::string, Bytes>;

float signed_log(float v) { return std::copysign(std::log1p(std::fabs(v)), v); }

uint8_t to_byte(float v) { return (uint8_t)std::lround(std::clamp(v, 0.0f, 255.0f)); }

// ================
// Codebooks
// ================

// 256 values by 1-D k-means over a strided subsample, seeded from equal-count slices.
std::vector<float> codebook_1d(const float* v, size_t n) {
    std::vector<float> s;
    const size_t stride = std::max<size_t>(1, n >> 20);
    s.reserve(n / stride + 1);
    for (size_t i = 0; i < n; i += stride)
        if (std::isfinite(v[i])) s.push_back(v[i]);
    std::sort(s.begin(), s.end());
    std::vector<float> cb(256, 0.0f);
    if (s.empty()) return cb;

    for (int k = 0; k < 256; k++) {
        size_t a = std::min(s.size() * k / 256, s.size() - 1);
        size_t b = std::max(s.size() * (k + 1) / 256, a + 1);
        double sum = 0;
        for (size_t i = a; i < b; i++) sum += s[i];
        cb[k] = (float)(sum / (double)(b - a));
    }
    for (int iter = 0; iter < 8; iter++) {
        double sum[256] = {};
        size_t cnt[256] = {};
        int k = 0;
        for (float x : s) {
            while (k < 255 && x > 0.5f * (cb[k] + cb[k + 1])) k++;
            sum[k] += x;
            cnt[k]++;
        }
        for (int j = 0; j < 256; j++)
            if (cnt[j]) cb[j] = (float)(sum[j] / (double)cnt[j]);
        std::sort(cb.begin(), cb.end());
    }
    return cb;
}

uint8_t nearest_code(const std::vector<float>& cb, float x) {
    auto it = std::lower_bound(cb.begin(), cb.end(), x);
    if (it == cb.begin()) return 0;
    if (it == cb.end()) return 255;
    const int hi = (int)(it - cb.begin());
    return (uint8_t)((x - cb[hi - 1] <= cb[hi] - x) ? hi - 1 : hi);
}

// ================
// Positions and order
// ================

struct Positions {
    float mins[3] = {};
    float maxs[3] = {};
    std::vector<uint16_t> q;  // [N, 3]
};

Positions quantize_positions(const SplatCloud& c) {
    const size_t n = (size_t)c.num;
    std::vector<float> lg(n * 3);
    for (size_t i = 0; i < n * 3; i++)
        lg[i] = std::isfinite(c.means[i]) ? signed_log(c.means[i]) : 0.0f;

    Positions p;
    p.q.resize(n * 3);
    for (int a = 0; a < 3; a++) {
        float lo = FLT_MAX, hi = -FLT_MAX;
        for (size_t i = 0; i < n; i++) {
            lo = std::min(lo, lg[i * 3 + a]);
            hi = std::max(hi, lg[i * 3 + a]);
        }
        p.mins[a] = lo;
        p.maxs[a] = hi;
        const float scale = hi > lo ? 65535.0f / (hi - lo) : 0.0f;
        for (size_t i = 0; i < n; i++)
            p.q[i * 3 + a] = (uint16_t)std::lround(std::clamp((lg[i * 3 + a] - lo) * scale, 0.0f, 65535.0f));
    }
    return p;
}

uint32_t spread3(uint32_t v) {
    v &= 0x3ff;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}

// Spatial order: neighbouring pixels are neighbouring Gaussians, which is what
// lets lossless WebP find redundancy in the position and colour images.
std::vector<uint32_t> morton_order(const Positions& p, size_t n) {
    std::vector<uint64_t> keys(n);
    for (size_t i = 0; i < n; i++) {
        const uint32_t m = spread3(p.q[i * 3] >> 6) | (spread3(p.q[i * 3 + 1] >> 6) << 1) |
                           (spread3(p.q[i * 3 + 2] >> 6) << 2);
        keys[i] = ((uint64_t)m << 32) | (uint64_t)i;
    }
    std::sort(keys.begin(), keys.end());
    std::vector<uint32_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = (uint32_t)(keys[i] & 0xffffffffu);
    return order;
}

void image_size(size_t n, int& w, int& h) {
    w = std::max(4, (int)std::ceil(std::sqrt((double)n) / 4.0) * 4);
    h = ((int)((n + (size_t)w - 1) / (size_t)w) + 3) / 4 * 4;
}

// ================
// Rotation
// ================

// Smallest-three: the largest component is dropped and made non-negative by
// negating the whole quaternion. Order is (w, x, y, z).
void encode_quat(const float* q, uint8_t* out) {
    float v[4];
    double len = 0;
    for (int k = 0; k < 4; k++) {
        v[k] = std::isfinite(q[k]) ? q[k] : 0.0f;
        len += (double)v[k] * v[k];
    }
    if (len <= 0) {
        v[0] = 1; v[1] = v[2] = v[3] = 0;
    } else {
        const float inv = (float)(1.0 / std::sqrt(len));
        for (float& x : v) x *= inv;
    }
    int m = 0;
    for (int k = 1; k < 4; k++)
        if (std::fabs(v[k]) > std::fabs(v[m])) m = k;
    const float sign = v[m] < 0 ? -1.0f : 1.0f;
    int o = 0;
    for (int k = 0; k < 4; k++) {
        if (k == m) continue;
        out[o++] = to_byte(255.0f * (v[k] * sign * kInvSqrt2 + 0.5f));
    }
    out[3] = (uint8_t)(252 + m);
}

// ================
// Higher-order SH palette
// ================

struct Palette {
    int bands = 0;
    int per_channel = 0;
    int count = 0;
    std::vector<float> centroids;  // [count, per_channel * 3], coefficient-major
    std::vector<uint16_t> labels;  // [N]
};

// `hint` seeds the bound: a neighbour's or last pass's winner makes the
// running-sum early-out discard most centroids within the first few dims.
int nearest_centroid(const float* x, const float* cent, int count, int d, int hint) {
    auto full = [&](int k) {
        const float* c = cent + (size_t)k * d;
        float s = 0;
        for (int j = 0; j < d; j++) {
            const float df = x[j] - c[j];
            s += df * df;
        }
        return s;
    };
    float best = full(hint);
    int best_k = hint;
    for (int k = 0; k < count; k++) {
        if (k == hint) continue;
        const float* c = cent + (size_t)k * d;
        float s = 0;
        int j = 0;
        bool pruned = false;
        for (; j + 8 <= d; j += 8) {
            for (int t = 0; t < 8; t++) {
                const float df = x[j + t] - c[j + t];
                s += df * df;
            }
            if (s >= best) { pruned = true; break; }
        }
        if (pruned) continue;
        for (; j < d; j++) {
            const float df = x[j] - c[j];
            s += df * df;
        }
        if (s < best) { best = s; best_k = k; }
    }
    return best_k;
}

Palette build_palette(const SplatCloud& c, int bands, const std::vector<uint32_t>& order) {
    const size_t n = (size_t)c.num;
    Palette pal;
    pal.bands = bands;
    pal.per_channel = bands * (bands + 2);
    const int d = pal.per_channel * 3;
    const size_t stride = (size_t)c.coeffs() * 3;
    auto row = [&](size_t i) { return c.features_sh.data() + i * stride; };

    int p = 64;
    while (p * 2 <= kMaxPalette && (size_t)p * 2 <= n / 32) p *= 2;
    p = (int)std::min<size_t>((size_t)p, n);
    pal.count = p;

    const size_t want = std::min(n, (size_t)std::max(p * 8, 32768));
    std::vector<size_t> sample(want);
    for (size_t s = 0; s < want; s++) sample[s] = s * n / want;

    std::mt19937 rng(12345);
    std::vector<size_t> pick(want);
    std::iota(pick.begin(), pick.end(), (size_t)0);
    std::shuffle(pick.begin(), pick.end(), rng);

    std::vector<float>& cent = pal.centroids;
    cent.resize((size_t)p * d);
    for (int k = 0; k < p; k++) std::copy_n(row(sample[pick[k]]), d, cent.begin() + (size_t)k * d);

    std::vector<int> assign(want, 0);
    std::vector<double> sum((size_t)p * d);
    std::vector<uint32_t> cnt((size_t)p);
    for (int iter = 0; iter < 8; iter++) {
#pragma omp parallel for schedule(static)
        for (long long s = 0; s < (long long)want; s++)
            assign[(size_t)s] = nearest_centroid(row(sample[(size_t)s]), cent.data(), p, d, assign[(size_t)s]);
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0u);
        for (size_t s = 0; s < want; s++) {
            const float* x = row(sample[s]);
            double* acc = sum.data() + (size_t)assign[s] * d;
            for (int j = 0; j < d; j++) acc[j] += x[j];
            cnt[(size_t)assign[s]]++;
        }
        for (int k = 0; k < p; k++) {
            float* dst = cent.data() + (size_t)k * d;
            if (cnt[(size_t)k] == 0) {
                std::copy_n(row(sample[rng() % want]), d, dst);
                continue;
            }
            for (int j = 0; j < d; j++) dst[j] = (float)(sum[(size_t)k * d + j] / cnt[(size_t)k]);
        }
    }

    pal.labels.resize(n);
#pragma omp parallel
    {
        int prev = 0;
#pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            const size_t g = order[(size_t)i];
            prev = nearest_centroid(row(g), cent.data(), p, d, prev);
            pal.labels[g] = (uint16_t)prev;
        }
    }
    return pal;
}

// ================
// Container
// ================

void put_floats(std::ostringstream& os, const float* v, size_t n) {
    os << '[';
    for (size_t i = 0; i < n; i++) os << (i ? "," : "") << v[i];
    os << ']';
}

Bytes encode_image(const Bytes& px, int w, int h, int channels, const char* name) {
    Bytes out;
    if (!encode_webp_lossless(px.data(), px.size(), w, h, channels, out))
        throw std::runtime_error(std::string("SOG: WebP encoding of ") + name + " failed");
    return out;
}

Bytes zip_entries(const std::vector<Entry>& files) {
    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, 1 << 20)) throw std::runtime_error("SOG: cannot start the archive");
    for (const Entry& e : files) {
        const bool text = e.first.size() > 5 && e.first.compare(e.first.size() - 5, 5, ".json") == 0;
        if (!mz_zip_writer_add_mem(&zip, e.first.c_str(), e.second.data(), e.second.size(),
                                   text ? (mz_uint)MZ_DEFAULT_LEVEL : (mz_uint)MZ_NO_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            throw std::runtime_error("SOG: cannot add " + e.first + " to the archive");
        }
    }
    void* buf = nullptr;
    size_t size = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip, &buf, &size)) {
        mz_zip_writer_end(&zip);
        throw std::runtime_error("SOG: cannot finish the archive");
    }
    Bytes out((const uint8_t*)buf, (const uint8_t*)buf + size);
    mz_free(buf);
    mz_zip_writer_end(&zip);
    return out;
}

}  // namespace

SogStats write_sog(const SplatCloud& c, const std::string& path) {
    if (c.num <= 0) throw std::runtime_error("SOG: the model has no splats");
    if ((uint64_t)c.num > 0xffffffffull) throw std::runtime_error("SOG: too many splats");
    const size_t n = (size_t)c.num;
    if (c.means.size() < n * 3 || c.quats.size() < n * 4 || c.scales.size() < n * 3 ||
        c.opacities.size() < n || c.features_dc.size() < n * 3)
        throw std::runtime_error("SOG: the model's arrays are shorter than its splat count");

    const int bands = std::min(c.sh_degree, 3);
    if (bands > 0 && c.features_sh.size() < n * (size_t)c.coeffs() * 3)
        throw std::runtime_error("SOG: the model's SH array is shorter than its splat count");

    const Positions pos = quantize_positions(c);
    const std::vector<uint32_t> order = morton_order(pos, n);
    int w = 0, h = 0;
    image_size(n, w, h);
    const size_t pixels = (size_t)w * h;

    Bytes means_l(pixels * 3), means_u(pixels * 3), scales_px(pixels * 3);
    Bytes quats_px(pixels * 4), sh0_px(pixels * 4);

    const std::vector<float> scale_cb = codebook_1d(c.scales.data(), n * 3);
    const std::vector<float> dc_cb = codebook_1d(c.features_dc.data(), n * 3);

    for (size_t i = 0; i < n; i++) {
        const size_t g = order[i];
        for (int a = 0; a < 3; a++) {
            const uint16_t q = pos.q[g * 3 + a];
            means_l[i * 3 + a] = (uint8_t)(q & 0xff);
            means_u[i * 3 + a] = (uint8_t)(q >> 8);
            scales_px[i * 3 + a] = nearest_code(scale_cb, c.scales[g * 3 + a]);
            sh0_px[i * 4 + a] = nearest_code(dc_cb, c.features_dc[g * 3 + a]);
        }
        encode_quat(&c.quats[g * 4], &quats_px[i * 4]);
        const float alpha = 1.0f / (1.0f + std::exp(-c.opacities[g]));
        sh0_px[i * 4 + 3] = to_byte(alpha * 255.0f);
    }

    std::vector<Entry> files;
    files.push_back({"means_l.webp", encode_image(means_l, w, h, 3, "means_l")});
    files.push_back({"means_u.webp", encode_image(means_u, w, h, 3, "means_u")});
    files.push_back({"scales.webp", encode_image(scales_px, w, h, 3, "scales")});
    files.push_back({"quats.webp", encode_image(quats_px, w, h, 4, "quats")});
    files.push_back({"sh0.webp", encode_image(sh0_px, w, h, 4, "sh0")});

    SogStats stats;
    stats.count = c.num;
    std::vector<float> shn_cb;
    if (bands > 0) {
        const Palette pal = build_palette(c, bands, order);
        stats.sh_bands = bands;
        stats.palette = pal.count;
        shn_cb = codebook_1d(pal.centroids.data(), pal.centroids.size());

        const int cw = 64 * pal.per_channel;
        const int ch = (pal.count + 63) / 64;
        Bytes cent_px((size_t)cw * ch * 3);
        for (int k = 0; k < pal.count; k++)
            for (int j = 0; j < pal.per_channel; j++)
                for (int a = 0; a < 3; a++) {
                    const size_t px = (size_t)(k / 64) * cw + (size_t)(k % 64) * pal.per_channel + j;
                    cent_px[px * 3 + a] =
                        nearest_code(shn_cb, pal.centroids[(size_t)k * pal.per_channel * 3 + j * 3 + a]);
                }
        Bytes label_px(pixels * 4);
        for (size_t i = 0; i < n; i++) {
            const uint16_t l = pal.labels[order[i]];
            label_px[i * 4] = (uint8_t)(l & 0xff);
            label_px[i * 4 + 1] = (uint8_t)(l >> 8);
            label_px[i * 4 + 3] = 255;
        }
        files.push_back({"shN_centroids.webp", encode_image(cent_px, cw, ch, 3, "shN_centroids")});
        files.push_back({"shN_labels.webp", encode_image(label_px, w, h, 4, "shN_labels")});
    }

    std::ostringstream js;
    js.imbue(std::locale::classic());
    js.precision(9);
    js << "{\"version\":2,\"asset\":{\"generator\":\"spirula\"},\"count\":" << n;
    js << ",\"means\":{\"mins\":";
    put_floats(js, pos.mins, 3);
    js << ",\"maxs\":";
    put_floats(js, pos.maxs, 3);
    js << ",\"files\":[\"means_l.webp\",\"means_u.webp\"]}";
    js << ",\"scales\":{\"codebook\":";
    put_floats(js, scale_cb.data(), 256);
    js << ",\"files\":[\"scales.webp\"]}";
    js << ",\"quats\":{\"files\":[\"quats.webp\"]}";
    js << ",\"sh0\":{\"codebook\":";
    put_floats(js, dc_cb.data(), 256);
    js << ",\"files\":[\"sh0.webp\"]}";
    if (bands > 0) {
        js << ",\"shN\":{\"count\":" << stats.palette << ",\"bands\":" << bands << ",\"codebook\":";
        put_floats(js, shn_cb.data(), 256);
        js << ",\"files\":[\"shN_centroids.webp\",\"shN_labels.webp\"]}";
    }
    js << "}";
    const std::string meta = js.str();
    files.insert(files.begin(), {"meta.json", Bytes(meta.begin(), meta.end())});

    const Bytes archive = zip_entries(files);
    const fs::path dest(path);
    const fs::path tmp = fs::path(path + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write((const char*)archive.data(), (std::streamsize)archive.size());
        if (!out) throw std::runtime_error("SOG: cannot write " + path);
    }
    replace_file(tmp, dest);
    stats.bytes = archive.size();
    return stats;
}

}  // namespace spirula
