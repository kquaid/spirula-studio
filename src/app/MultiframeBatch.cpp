#include "app/MultiframeBatch.h"

#include "checkpoint/SogWriter.h"
#include "checkpoint/SplatPly.h"
#include "core/AtomicFile.h"
#include "i18n/catalog/Log.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace spirula {
namespace {

namespace fs = std::filesystem;
namespace lmsg = spirula::i18n::msg::log;

std::string lfmt(const i18n::Msg& m, std::initializer_list<i18n::Arg> a) {
    return i18n::format(m, a);
}

std::string frame_file(int n, const char* ext) {
    char name[48];
    std::snprintf(name, sizeof name, "frame_%06d.%s", n, ext);
    return name;
}

// SOG encoding is host-only and takes longer than a short frame trains, so it
// runs beside the next frame. Two jobs wait at most; submit() blocks beyond that.
class SogExporter {
public:
    explicit SogExporter(TrainerSession& s) : _s(s), _thread([this] { run(); }) {}
    ~SogExporter() { finish(); }

    void submit(fs::path ply, fs::path sog) {
        std::unique_lock<std::mutex> lk(_mu);
        _room.wait(lk, [&] { return _jobs.size() < 2; });
        _jobs.push_back({std::move(ply), std::move(sog)});
        _work.notify_one();
    }

    bool pending() {
        std::lock_guard<std::mutex> lk(_mu);
        return !_jobs.empty() || _active;
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(_mu);
            _closed = true;
        }
        _work.notify_all();
        if (_thread.joinable()) _thread.join();
    }

private:
    struct Job {
        fs::path ply, sog;
    };

    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(_mu);
                _work.wait(lk, [&] { return _closed || !_jobs.empty(); });
                if (_jobs.empty()) return;
                job = std::move(_jobs.front());
                _jobs.pop_front();
                _active = true;
                _room.notify_all();
            }
            convert(job);
            std::lock_guard<std::mutex> lk(_mu);
            _active = false;
        }
    }

    void convert(const Job& job) {
        try {
            const auto t0 = std::chrono::steady_clock::now();
            const SplatCloud cloud = read_splat_ply(job.ply.string(), true);
            const SogStats stats = write_sog(cloud, job.sog.string());
            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            char mb[32], s[32];
            std::snprintf(mb, sizeof mb, "%.1f", (double)stats.bytes / (1024.0 * 1024.0));
            std::snprintf(s, sizeof s, "%.1f", secs);
            _s.log(lfmt(lmsg::sog_saved, {job.sog.string(), mb, s}));
        } catch (const std::exception& e) {
            _s.log(lfmt(lmsg::sog_failed, {e.what()}));
        }
    }

    TrainerSession& _s;
    std::mutex _mu;
    std::condition_variable _work, _room;
    std::deque<Job> _jobs;
    bool _closed = false;
    bool _active = false;
    std::thread _thread;   // last: starts once the rest exists
};

void check_frame_files(const TrainerSession& s, int n) {
    for (const auto* files : {&s.ds.image_filenames, &s.ds.depth_filenames, &s.ds.normal_filenames})
        for (const std::string& f : *files)
            if (!f.empty() && !fs::exists(f))
                throw std::runtime_error(lfmt(lmsg::mf_err_file_missing, {(long long)n, f}));
}

}  // namespace

int multiframe_frame_number(const std::string& name) {
    if (name.empty() || name.size() > 9) return -1;
    for (char c : name)
        if (c < '0' || c > '9') return -1;
    return std::stoi(name);
}

std::vector<int> multiframe_frames(const std::string& root, int first, int last) {
    auto folder = [&](int n) { return fs::path(root) / std::to_string(n); };
    auto exists = [&](int n) {
        std::error_code ec;
        return fs::is_directory(folder(n), ec);
    };
    std::vector<int> frames;
    if (last >= 0) {
        for (int n = first; n <= last; n++) {
            if (!exists(n))
                throw std::runtime_error(lfmt(lmsg::mf_err_frame_missing, {folder(n).string()}));
            frames.push_back(n);
        }
    } else {
        for (int n = first; exists(n); n++) frames.push_back(n);
    }
    if (frames.empty())
        throw std::runtime_error(lfmt(lmsg::mf_err_frame_missing, {folder(first).string()}));
    return frames;
}

TrainConfig multiframe_refine_config(const TrainConfig& base, const MultiframeSettings& m) {
    TrainConfig c = base;
    c.num_iterations = std::max(1, m.frame_iterations);
    c.resume.clear();
    c.init_ply.clear();
    c.init_ply_add_points = false;
    c.steps_per_save = 0;
    c.save_sog = false;
    // No growth, relocation or MCMC noise: the count and every splat's index stay put.
    c.refine_every = 0;
    c.noise_lr = 0.0f;
    c.noise_lr_final = 0.0f;
    // Every step-keyed ramp would otherwise restart inside a 100-step frame.
    c.sh_degree_warmup_every = 0;
    c.progressive_resolution = false;
    const float k = std::max(m.lr_scale, 0.0f);
    c.means_lr *= k;
    c.means_lr_final = c.means_lr;
    c.scales_lr *= k;
    c.scales_lr_final = c.scales_lr;
    c.quats_lr *= k;
    c.opacities_lr *= k;
    c.features_dc_lr *= k;
    c.features_sh_lr *= k;
    c.features_sb_lr *= k;
    return c;
}

void run_multiframe(TrainerSession& s, const MultiframeSettings& m,
                    const MultiframeHooks& h, const TrainerCallbacks& cb) {
    fs::path opened = fs::absolute(s.cfg.data).lexically_normal();
    if (!opened.has_filename()) opened = opened.parent_path();
    const int opened_n = multiframe_frame_number(opened.filename().string());
    if (opened_n < 0)
        throw std::runtime_error(lfmt(lmsg::mf_err_folder_name, {opened.filename().string()}));
    const fs::path root = opened.parent_path();

    const std::vector<int> frames =
        multiframe_frames(root.string(), m.first_frame >= 0 ? m.first_frame : opened_n, m.last_frame);
    const int total = (int)frames.size();
    const fs::path out = m.output_dir.empty() ? root / "multiframe_out" : fs::path(m.output_dir);
    fs::create_directories(out / "ply");
    fs::create_directories(out / "sog");
    s.out_dir_override = (out / "run").string();
    s.log(lfmt(lmsg::mf_batch_start, {(long long)total, (long long)frames.front(),
                                      (long long)frames.back(), out.string()}));

    const TrainConfig refine = multiframe_refine_config(s.cfg, m);
    s.cfg.save_sog = false;   // each frame is exported here, not by train()
    SogExporter exporter(s);
    fs::path current = opened;
    int done = 0;
    for (int i = 0; i < total; i++) {
        const int n = frames[(size_t)i];
        const fs::path dir = root / std::to_string(n);
        if (dir != current) {
            s.rebase_dataset(current.string(), dir.string());
            current = dir;
        }
        check_frame_files(s, n);
        s.log(lfmt(lmsg::mf_frame_start, {(long long)n, (long long)(i + 1), (long long)total}));
        {
            std::lock_guard<std::mutex> lk(s.engine_mutex);
            s.start_step = 0;
            s.cur_step = 0;
            s.setup_engine();
            if (h.engine_ready) h.engine_ready(n, i, total);
        }
        if (i == 0) {
            std::error_code ec;
            fs::copy_file(out / "run" / "scene_transform.json", out / "scene_transform.json",
                          fs::copy_options::overwrite_existing, ec);
        }

        s.train(cb);
        if (s.stop_requested.load()) break;

        const fs::path tmp = out / "tmp";
        SeedSplats next;
        {
            std::lock_guard<std::mutex> lk(s.engine_mutex);
            fs::remove_all(tmp);
            fs::create_directories(tmp);
            engine_save_checkpoint(tmp.string(), false, s.cur_step.load(),
                                   s.cfg.background_export_height);
            next = s.capture_splats();
        }
        const fs::path ply = out / "ply" / frame_file(n, "ply");
        replace_file(tmp / "splat.ply", ply);
        fs::remove_all(tmp);
        exporter.submit(ply, out / "sog" / frame_file(n, "sog"));
        done++;
        s.log(lfmt(lmsg::mf_frame_done, {(long long)n, ply.string()}));
        if (h.frame_done) h.frame_done(n, i, total);

        if (i == 0) {
            s.cfg = refine;
            s.write_config_json = false;
        }
        s.cfg.data = current.string();
        s.carried_splats = std::move(next);
    }

    if (exporter.pending()) s.log(lmsg::mf_sog_waiting.get());
    exporter.finish();
    s.log(done == total ? lfmt(lmsg::mf_batch_done, {(long long)done})
                        : lfmt(lmsg::mf_batch_stopped, {(long long)done, (long long)total}));
}

}  // namespace spirula
