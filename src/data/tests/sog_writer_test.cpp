#include "checkpoint/SogWriter.h"

#include "external/miniz.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#ifdef SS_TEST_WEBP_DECODER
#include <webp/decode.h>
#endif

namespace {
int failures = 0;

void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

spirula::SplatCloud make_cloud(int n, int sh_degree) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> pos(-5.0f, 5.0f), logs(-6.0f, -1.0f);
    std::uniform_real_distribution<float> dc(-2.0f, 2.0f), logit(-4.0f, 4.0f);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f), centre(-0.5f, 0.5f);
    std::normal_distribution<float> noise(0.0f, 0.002f);

    spirula::SplatCloud c;
    c.num = n;
    c.sh_degree = sh_degree;
    c.means.resize((size_t)n * 3);
    c.quats.resize((size_t)n * 4);
    c.scales.resize((size_t)n * 3);
    c.opacities.resize((size_t)n);
    c.features_dc.resize((size_t)n * 3);
    for (auto& v : c.means) v = pos(rng);
    for (auto& v : c.quats) v = unit(rng);
    for (auto& v : c.scales) v = logs(rng);
    for (auto& v : c.opacities) v = logit(rng);
    for (auto& v : c.features_dc) v = dc(rng);

    const int coeffs = (int)c.coeffs();
    if (coeffs > 0) {
        std::vector<std::vector<float>> base(16, std::vector<float>((size_t)coeffs * 3));
        for (auto& b : base)
            for (auto& v : b) v = centre(rng);
        c.features_sh.resize((size_t)n * coeffs * 3);
        for (int i = 0; i < n; i++)
            for (size_t j = 0; j < (size_t)coeffs * 3; j++)
                c.features_sh[(size_t)i * coeffs * 3 + j] = base[(size_t)i % 16][j] + noise(rng);
    }
    return c;
}

#ifdef SS_TEST_WEBP_DECODER
struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;
    const uint8_t* at(int x, int y) const { return &rgba[((size_t)y * w + x) * 4]; }
};

bool extract(mz_zip_archive& zip, const char* name, std::vector<uint8_t>& out) {
    size_t size = 0;
    void* p = mz_zip_reader_extract_file_to_heap(&zip, name, &size, 0);
    if (!p) return false;
    out.assign((const uint8_t*)p, (const uint8_t*)p + size);
    mz_free(p);
    return true;
}

bool decode_image(mz_zip_archive& zip, const char* name, Image& img) {
    std::vector<uint8_t> bytes;
    if (!extract(zip, name, bytes)) return false;
    uint8_t* px = WebPDecodeRGBA(bytes.data(), bytes.size(), &img.w, &img.h);
    if (!px) return false;
    img.rgba.assign(px, px + (size_t)img.w * img.h * 4);
    WebPFree(px);
    return true;
}

// Pulls the first float array that follows `key` out of meta.json.
std::vector<double> json_array(const std::string& s, const std::string& key, size_t from = 0) {
    std::vector<double> out;
    size_t p = s.find(key, from);
    if (p == std::string::npos) return out;
    p = s.find('[', p);
    const size_t e = s.find(']', p);
    const char* c = s.c_str() + p + 1;
    while (c < s.c_str() + e) {
        char* next = nullptr;
        out.push_back(std::strtod(c, &next));
        c = next;
        if (*c == ',') c++;
    }
    return out;
}

void roundtrip(int n, int sh_degree) {
    const spirula::SplatCloud c = make_cloud(n, sh_degree);
    const auto path = std::filesystem::temp_directory_path() / ("sog_writer_test_" + std::to_string(sh_degree) + ".sog");
    const spirula::SogStats stats = spirula::write_sog(c, path.string());
    check(stats.count == n && stats.bytes > 0, "stats reflect the write");

    mz_zip_archive zip{};
    check(mz_zip_reader_init_file(&zip, path.string().c_str(), 0), "archive opens");
    std::vector<uint8_t> meta_bytes;
    check(extract(zip, "meta.json", meta_bytes), "meta.json is in the archive");
    const std::string meta(meta_bytes.begin(), meta_bytes.end());
    check(meta.find("\"version\":2") != std::string::npos, "version 2");

    Image l, u, sc, qt, s0;
    check(decode_image(zip, "means_l.webp", l) && decode_image(zip, "means_u.webp", u) &&
              decode_image(zip, "scales.webp", sc) && decode_image(zip, "quats.webp", qt) &&
              decode_image(zip, "sh0.webp", s0),
          "all property images decode");
    check(l.w * l.h >= n && l.w % 4 == 0 && l.h % 4 == 0, "image holds every splat, in multiples of four");

    const auto mins = json_array(meta, "\"mins\"");
    const auto maxs = json_array(meta, "\"maxs\"");
    const auto scale_cb = json_array(meta, "\"scales\"");
    const auto dc_cb = json_array(meta, "\"sh0\"");
    check(mins.size() == 3 && maxs.size() == 3 && scale_cb.size() == 256 && dc_cb.size() == 256,
          "meta arrays have the specified sizes");
    if (failures) { mz_zip_reader_end(&zip); return; }

    struct Decoded { double p[3]; double q[4]; double s[3]; double dc[3]; double alpha; int x, y; };
    std::vector<Decoded> dec((size_t)n);
    for (int i = 0; i < n; i++) {
        const int x = i % l.w, y = i / l.w;
        Decoded& d = dec[(size_t)i];
        d.x = x; d.y = y;
        for (int a = 0; a < 3; a++) {
            const double qv = (double)((u.at(x, y)[a] << 8) | l.at(x, y)[a]) / 65535.0;
            const double nv = mins[a] + (maxs[a] - mins[a]) * qv;
            d.p[a] = (nv < 0 ? -1.0 : 1.0) * (std::exp(std::fabs(nv)) - 1.0);
            d.s[a] = std::exp(scale_cb[sc.at(x, y)[a]]);
            d.dc[a] = dc_cb[s0.at(x, y)[a]];
        }
        d.alpha = s0.at(x, y)[3] / 255.0;
        const uint8_t* q = qt.at(x, y);
        const int mode = q[3] - 252;
        check(mode >= 0 && mode <= 3, "quaternion mode is 252..255");
        double comp[3];
        for (int k = 0; k < 3; k++) comp[k] = (q[k] / 255.0 - 0.5) * 2.0 / std::sqrt(2.0);
        const double t = comp[0] * comp[0] + comp[1] * comp[1] + comp[2] * comp[2];
        const double om = std::sqrt(std::max(0.0, 1.0 - t));
        int o = 0;
        for (int k = 0; k < 4; k++) d.q[k] = (k == mode) ? om : comp[o++];
    }

    std::vector<int> match((size_t)n, -1);
    double worst_pos = 0, worst_scale = 0, worst_dc = 0, worst_alpha = 0, worst_quat = 1;
    for (int g = 0; g < n; g++) {
        int best = -1;
        double best_d = 1e30;
        for (int j = 0; j < n; j++) {
            double dd = 0;
            for (int a = 0; a < 3; a++) {
                const double df = dec[(size_t)j].p[a] - c.means[(size_t)g * 3 + a];
                dd += df * df;
            }
            if (dd < best_d) { best_d = dd; best = j; }
        }
        match[(size_t)g] = best;
        const Decoded& d = dec[(size_t)best];
        worst_pos = std::max(worst_pos, std::sqrt(best_d));
        for (int a = 0; a < 3; a++) {
            worst_scale = std::max(worst_scale, std::fabs(std::log(d.s[a]) - c.scales[(size_t)g * 3 + a]));
            worst_dc = std::max(worst_dc, std::fabs(d.dc[a] - c.features_dc[(size_t)g * 3 + a]));
        }
        worst_alpha = std::max(worst_alpha,
                               std::fabs(d.alpha - 1.0 / (1.0 + std::exp(-(double)c.opacities[(size_t)g]))));
        double dot = 0, len = 0;
        for (int k = 0; k < 4; k++) {
            dot += d.q[k] * c.quats[(size_t)g * 4 + k];
            len += (double)c.quats[(size_t)g * 4 + k] * c.quats[(size_t)g * 4 + k];
        }
        worst_quat = std::min(worst_quat, std::fabs(dot) / std::sqrt(len));
    }
    std::printf("n=%d sh=%d bytes=%zu pos=%.2e scale=%.3f dc=%.3f alpha=%.4f quat_dot=%.5f\n", n, sh_degree,
                stats.bytes, worst_pos, worst_scale, worst_dc, worst_alpha, worst_quat);
    check(worst_pos < 1e-3, "positions within 1e-3");
    check(worst_scale < 0.03, "scales within 0.03 in log space");
    check(worst_dc < 0.03, "DC within 0.03");
    check(worst_alpha < 0.005, "opacity within one 8-bit step");
    check(worst_quat > 0.999, "rotations within a fraction of a degree");

    const bool has_shn = meta.find("\"shN\"") != std::string::npos;
    check(has_shn == (sh_degree > 0), "shN present exactly when the cloud has SH");
    if (has_shn) {
        Image cent, lab;
        check(decode_image(zip, "shN_centroids.webp", cent) && decode_image(zip, "shN_labels.webp", lab),
              "shN images decode");
        const int per = sh_degree * (sh_degree + 2);
        check(cent.w == 64 * per, "centroid texture is 64 entries wide");
        const size_t at = meta.find("\"shN\"");
        const auto cb = json_array(meta, "\"codebook\"", at);
        check(cb.size() == 256, "shN codebook has 256 entries");
        double worst = 0;
        const size_t stride = (size_t)c.coeffs() * 3;
        for (int g = 0; g < n && cb.size() == 256; g++) {
            const Decoded& d = dec[(size_t)match[(size_t)g]];
            const uint8_t* lp = lab.at(d.x, d.y);
            const int label = lp[0] + (lp[1] << 8);
            for (int k = 0; k < per; k++) {
                const uint8_t* px = cent.at((label % 64) * per + k, label / 64);
                for (int a = 0; a < 3; a++)
                    worst = std::max(worst, std::fabs(cb[px[a]] - c.features_sh[(size_t)g * stride + k * 3 + a]));
            }
        }
        std::printf("  shN worst error %.4f\n", worst);
        check(worst < 0.05, "SH palette reproduces clustered coefficients within 0.05");
    }
    mz_zip_reader_end(&zip);
    std::filesystem::remove(path);
}
#endif

void rejects_bad_input() {
    spirula::SplatCloud empty;
    bool threw = false;
    try { spirula::write_sog(empty, (std::filesystem::temp_directory_path() / "sog_empty.sog").string()); }
    catch (const std::exception&) { threw = true; }
    check(threw, "an empty cloud is refused");

    spirula::SplatCloud c = make_cloud(100, 0);
    c.scales.resize(10);
    threw = false;
    try { spirula::write_sog(c, (std::filesystem::temp_directory_path() / "sog_short.sog").string()); }
    catch (const std::exception&) { threw = true; }
    check(threw, "short arrays are refused");
}
}  // namespace

int main() {
    rejects_bad_input();
#ifdef SS_TEST_WEBP_DECODER
    roundtrip(2000, 0);
    roundtrip(2000, 3);
    roundtrip(1000, 1);
#else
    std::printf("no WebP decoder; round-trip skipped\n");
#endif
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("sog_writer_test: ok\n");
    return 0;
}
