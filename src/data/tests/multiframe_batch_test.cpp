#include "app/MultiframeBatch.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {
int failures = 0;

void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

template <class F>
bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

void frame_numbers() {
    using spirula::multiframe_frame_number;
    check(multiframe_frame_number("0") == 0, "0 is frame 0");
    check(multiframe_frame_number("12") == 12, "12 is frame 12");
    check(multiframe_frame_number("007") == 7, "leading zeros still parse");
    check(multiframe_frame_number("") == -1, "empty is not a frame");
    check(multiframe_frame_number("anim") == -1, "words are not frames");
    check(multiframe_frame_number("1a") == -1, "mixed is not a frame");
    check(multiframe_frame_number("1234567890") == -1, "too long to be a frame");
}

void frame_discovery() {
    const fs::path root = fs::temp_directory_path() / "multiframe_batch_test";
    fs::remove_all(root);
    for (int n : {0, 1, 2, 4}) fs::create_directories(root / std::to_string(n));

    const auto all = spirula::multiframe_frames(root.string(), 0, -1);
    check(all.size() == 3 && all.front() == 0 && all.back() == 2, "open-ended run stops at the first gap");
    const auto from1 = spirula::multiframe_frames(root.string(), 1, -1);
    check(from1.size() == 2 && from1.front() == 1, "open-ended run starts where asked");
    const auto range = spirula::multiframe_frames(root.string(), 1, 2);
    check(range.size() == 2 && range.back() == 2, "an explicit range is honoured");
    check(throws([&] { spirula::multiframe_frames(root.string(), 0, 4); }),
          "a gap inside an explicit range stops the batch");
    check(throws([&] { spirula::multiframe_frames(root.string(), 9, -1); }),
          "a missing first frame stops the batch");
    check(throws([&] { spirula::multiframe_frames(root.string(), 3, 1); }),
          "an inverted range runs nothing and is refused");
    fs::remove_all(root);
}

void refine_config() {
    TrainConfig base;
    base.num_iterations = 30000;
    base.refine_every = 100;
    base.noise_lr = 160.0f;
    base.noise_lr_final = 1.6f;
    base.steps_per_save = 2000;
    base.save_sog = true;
    base.progressive_resolution = true;
    base.sh_degree_warmup_every = 1000;
    base.init_ply = "x.ply";

    spirula::MultiframeSettings m;
    m.frame_iterations = 100;
    m.lr_scale = 0.5f;
    const TrainConfig c = spirula::multiframe_refine_config(base, m);

    check(c.num_iterations == 100, "frames run the chosen iteration count");
    check(c.refine_every == 0, "growth and relocation are off");
    check(c.noise_lr == 0.0f && c.noise_lr_final == 0.0f, "MCMC noise is off");
    check(c.steps_per_save == 0 && !c.save_sog, "the batch saves frames itself");
    check(!c.progressive_resolution && c.sh_degree_warmup_every == 0, "no step-keyed ramps");
    check(c.init_ply.empty(), "frames seed from the previous frame, not a file");
    check(c.means_lr == base.means_lr * 0.5f, "rates are scaled");
    check(c.means_lr_final.has_value() && *c.means_lr_final == c.means_lr, "means rate does not decay");
    check(c.scales_lr_final.has_value() && *c.scales_lr_final == c.scales_lr, "scales rate does not decay");
    check(base.refine_every == 100 && base.num_iterations == 30000, "the first frame's config is untouched");
}

void rebase() {
    spirula::TrainerSession s;
    const fs::path root = fs::temp_directory_path() / "mf_rebase";
    s.cfg.data = (root / "0").string();
    s.ds.image_filenames = {(root / "0" / "images" / "a.png").string()};
    s.ds.depth_filenames = {(root / "0" / "depths" / "a.png").string(), ""};
    s.ds.normal_filenames = {(root / "0" / "normals" / "a.png").string()};
    s.rebase_dataset((root / "0").string(), (root / "5").string());
    check(s.ds.image_filenames[0] == (root / "5" / "images" / "a.png").string(), "images move to the new frame");
    check(s.ds.depth_filenames[0] == (root / "5" / "depths" / "a.png").string(), "depths move too");
    check(s.ds.depth_filenames[1].empty(), "a camera without a depth file stays without");
    check(s.ds.normal_filenames[0] == (root / "5" / "normals" / "a.png").string(), "normals move too");
    check(s.cfg.data == (root / "5").string(), "cfg.data follows the frame");
    check(throws([&] { s.rebase_dataset((root / "0").string(), (root / "6").string()); }),
          "paths no longer under the old folder are refused");
}
}  // namespace

int main() {
    frame_numbers();
    frame_discovery();
    refine_config();
    rebase();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("multiframe_batch_test: ok\n");
    return 0;
}
