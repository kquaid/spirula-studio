// TrainRunner.cpp -- see TrainRunner.h.

#include "app/gui/TrainRunner.h"

#include "backend/api/BackendRuntime.h"
#include "engine/Engine.h"
#include "i18n/catalog/Gui.h"
#include "i18n/catalog/Log.h"
#ifdef SS_HAVE_LPIPS
#include "app/gui/FetchSource.h"
#include "lpips/Fetch.h"
#endif

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace gui {

using spirula::TrainerSession;

std::string TrainRunner::fetch_line() {
    std::lock_guard<std::mutex> lk(_mu);
    return _fetch_line;
}

float TrainRunner::fetch_progress() {
    std::lock_guard<std::mutex> lk(_mu);
    return _fetch_progress;
}

void TrainRunner::fetch_eval_models(const TrainConfig& cfg) {
#ifdef SS_HAVE_LPIPS
    if (!spirula::lpips_wanted(cfg) || !cfg.lpips_checkpoints.empty()) return;
    namespace gmsg = spirula::i18n::msg::gui;
    for (const nn::FetchFile& f : lpips::missing_files({lpips::Net::Alex, lpips::Net::Vgg})) {
        PendingDownload d = pending_download(f);
        d.sha256 = f.sha256;
        FileDownload dl;
        dl.start(d);
        while (dl.state() == FileDownload::State::Running) {
            if (_session && _session->stop_requested.load()) dl.cancel();
            {
                std::lock_guard<std::mutex> lk(_mu);
                _fetch_line = spirula::i18n::format(gmsg::fetching_lpips, {f.file, dl.status()});
                _fetch_progress = dl.progress();
            }
            for (const std::string& line : dl.drain_log()) push_log(line);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        for (const std::string& line : dl.drain_log()) push_log(line);
        {
            std::lock_guard<std::mutex> lk(_mu);
            _fetch_line.clear();
            _fetch_progress = -1.0f;
        }
        if (dl.state() != FileDownload::State::Done)
            throw std::runtime_error(
                spirula::i18n::format(gmsg::lpips_fetch_failed, {f.file, dl.status()}));
    }
#else
    (void)cfg;
#endif
}

void TrainRunner::push_log(const std::string& s) {
    std::lock_guard<std::mutex> lk(_mu);
    _log.push_back(s);
    if (_log.size() > 5000) _log.erase(_log.begin(), _log.begin() + 1000);
}

std::vector<std::string> TrainRunner::drain_log() {
    std::lock_guard<std::mutex> lk(_mu);
    std::vector<std::string> out;
    out.swap(_log);
    return out;
}

std::string TrainRunner::error() {
    std::lock_guard<std::mutex> lk(_mu);
    return _error;
}

spirula::TrainerProgress TrainRunner::latest_progress() {
    std::lock_guard<std::mutex> lk(_mu);
    return _latest;
}

spirula::EvalProgress TrainRunner::latest_eval_progress() {
    std::lock_guard<std::mutex> lk(_mu);
    return _eval;
}

double TrainRunner::avg_latency_locked() const {
    if (_latencies.empty()) return -1.0;
    double sum = 0.0;
    for (double v : _latencies) sum += v;
    return sum / (double)_latencies.size();
}

double TrainRunner::avg_step_latency() {
    std::lock_guard<std::mutex> lk(_mu);
    return avg_latency_locked();
}

double TrainRunner::eta_seconds() {
    return _session ? _session->eta_seconds() : -1.0;
}

double TrainRunner::elapsed_seconds() {
    if (!_session) return -1.0;
    return _session->elapsed_seconds();
}

void TrainRunner::get_metrics(std::vector<MetricPoint>& out) {
    std::lock_guard<std::mutex> lk(_mu);
    out = _metrics;
}

void TrainRunner::join_worker() {
    if (_worker.joinable()) _worker.join();
}

void TrainRunner::set_paused(bool p) {
    if (_session) _session->paused = p;
}

bool TrainRunner::paused() const {
    return _session && _session->paused.load();
}

void TrainRunner::request_stop(bool save) {
    if (!_session) return;
    if (!save) _session->save_on_stop = false;
    _session->stop_requested = true;
    _data_cv.notify_all();   // the step loop may be parked on a file error
}

void TrainRunner::shutdown() {
    request_stop();
    join_worker();
    if (_web_viewer) { _web_viewer->stop(); _web_viewer.reset(); }
}

void TrainRunner::release_engine() {
    shutdown();
    _engine_ready = false;
    _session.reset();
    engine_reset();
}

void TrainRunner::load_dataset(const TrainConfig& cfg, const std::string& preset) {
    shutdown();
    _engine_ready = false;
    {
        std::lock_guard<std::mutex> lk(_mu);
        _error.clear();
    }
    _session.reset(new TrainerSession());
    _session->cfg = cfg;
    _session->preset = preset;
    _session->log_fn = [this](const std::string& s) { push_log(s); };
    _mf_active = false;
    _phase = Phase::Loading;
    TrainerSession* s = _session.get();
    _worker = std::thread([this, s] {
        try {
            s->check_config();
            s->load_dataset();
            _phase = Phase::Ready;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(_mu);
            _error = e.what();
            _phase = Phase::LoadError;
        }
    });
}

void TrainRunner::start_training(const TrainConfig& cfg, const std::string& preset) {
    shutdown();
    _engine_ready = false;
    {
        std::lock_guard<std::mutex> lk(_mu);
        _error.clear();
        _latest = {};
        _eval = {};
        _latencies.clear();
        _metrics.clear();
    }
    _session.reset(new TrainerSession());
    _session->cfg = cfg;
    _session->preset = preset;
    _session->log_fn = [this](const std::string& s) { push_log(s); };
    _mf_active = false;
    _phase = Phase::Preparing;
    TrainerSession* s = _session.get();
    _worker = std::thread([this, s] {
        try {
#ifndef SS_BACKEND_VULKAN
            // The CUDA runtime's current device is per thread; this worker
            // does all the engine work, so it re-applies the process-wide
            // selection before the first driver call.
            if (!backend::device_bind())
                throw std::runtime_error(
                    "could not make the selected GPU current on the training "
                    "thread; restart the application or choose another GPU");
#endif
            s->check_config();
            fetch_eval_models(s->cfg);
            s->load_dataset();
            s->setup_engine();
            s->viewer_base_camera_size = viewer_upload_cameras(s->post);
            viewer_upload_grid(s->post);
            _engine_ready = true;

            start_web_viewer(*s);

            _phase = Phase::Training;
            spirula::TrainerCallbacks cb;
            cb.on_data_error = [this](const std::string& what) {
                return await_data_decision(what);
            };
            cb.on_step = [this, metrics_ready = false](const spirula::TrainerProgress& p) mutable {
                record_step(p, metrics_ready);
            };
            s->train(cb);
            if (!s->stop_requested.load() && s->cfg.eval_mode != "all") {
                _phase = Phase::Evaluating;
                s->eval([this](const spirula::EvalProgress& p) {
                    std::lock_guard<std::mutex> lk(_mu);
                    _eval = p;
                });
            }
            _phase = Phase::Done;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(_mu);
            _error = e.what();
            _phase = Phase::TrainError;
        }
    });
}

void TrainRunner::record_step(const spirula::TrainerProgress& p, bool& metrics_ready) {
    std::lock_guard<std::mutex> lk(_mu);
    _latest = p;
    _latencies.push_back(p.step_latency);
    if (_latencies.size() > 100) _latencies.pop_front();
    // Scalar readbacks lag one step; the first callback has no metrics yet.
    if (!metrics_ready) {
        metrics_ready = true;
        return;
    }
    MetricPoint m;
    m.step = p.step;
    auto get = [&](const char* k) {
        auto it = p.losses.find(k);
        return it == p.losses.end() ? 0.0f : it->second;
    };
    m.psnr = get("psnr");
    m.ssim = get("ssim");
    m.rgb_loss = get("rgb_loss");
    m.num_splats = (float)p.num_splats;
    _metrics.push_back(m);
}

// Optional web viewer alongside the native viewport.
void TrainRunner::start_web_viewer(spirula::TrainerSession& s) {
    if (s.cfg.disable_viewer) return;
    _web_viewer.reset(new ViewerServer());
    _web_viewer->start("0.0.0.0", s.cfg.viewer_port, s.make_viewer_config(),
                       s.make_viewer_hooks(), s.post);
    push_log(spirula::i18n::format(spirula::i18n::msg::log::web_viewer_at,
                                   {(long long)s.cfg.viewer_port}));
}

bool TrainRunner::multiframe_progress(int& frame, int& index, int& total) const {
    if (!_mf_active.load()) return false;
    frame = _mf_frame.load();
    index = _mf_index.load();
    total = _mf_total.load();
    return true;
}

void TrainRunner::start_multiframe(const TrainConfig& cfg, const std::string& preset,
                                   const spirula::MultiframeSettings& settings) {
    shutdown();
    _engine_ready = false;
    {
        std::lock_guard<std::mutex> lk(_mu);
        _error.clear();
        _latest = {};
        _eval = {};
        _latencies.clear();
        _metrics.clear();
    }
    _session.reset(new TrainerSession());
    _session->cfg = cfg;
    _session->preset = preset;
    _session->log_fn = [this](const std::string& s) { push_log(s); };
    _mf_active = true;
    _mf_total = 0;
    _phase = Phase::Preparing;
    TrainerSession* s = _session.get();
    _worker = std::thread([this, s, settings] {
        try {
#ifndef SS_BACKEND_VULKAN
            if (!backend::device_bind())
                throw std::runtime_error(
                    "could not make the selected GPU current on the training "
                    "thread; restart the application or choose another GPU");
#endif
            s->check_config();
            fetch_eval_models(s->cfg);
            s->load_dataset();

            bool first = true;
            spirula::MultiframeHooks hooks;
            hooks.engine_ready = [this, s, &first](int frame, int index, int total) {
                s->viewer_base_camera_size = viewer_upload_cameras(s->post);
                viewer_upload_grid(s->post);
                _mf_frame = frame;
                _mf_index = index;
                _mf_total = total;
                {
                    // Each frame's plots start over; the batch's own progress is the frame count.
                    std::lock_guard<std::mutex> lk(_mu);
                    _latest = {};
                    _latencies.clear();
                    _metrics.clear();
                }
                if (first) {
                    first = false;
                    _engine_ready = true;
                    _phase = Phase::Training;
                }
            };
            spirula::TrainerCallbacks cb;
            cb.on_data_error = [this](const std::string& what) { return await_data_decision(what); };
            cb.on_step = [this, metrics_ready = false, web = false, s](
                             const spirula::TrainerProgress& p) mutable {
                record_step(p, metrics_ready);
                if (!web) {
                    web = true;
                    start_web_viewer(*s);
                }
            };
            spirula::run_multiframe(*s, settings, hooks, cb);
            _phase = Phase::Done;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(_mu);
            _error = e.what();
            _phase = Phase::TrainError;
        }
    });
}

bool TrainRunner::await_data_decision(const std::string& what) {
    std::unique_lock<std::mutex> lk(_data_mu);
    _data_err    = what;
    _data_answer = 0;
    _data_cv.wait(lk, [&]{
        return _data_answer != 0 ||
               (_session && _session->stop_requested.load());
    });
    _data_err.clear();
    return _data_answer == 1;
}

std::string TrainRunner::data_error() {
    std::lock_guard<std::mutex> lk(_data_mu);
    return _data_err;
}

void TrainRunner::resolve_data_error(bool retry) {
    {
        std::lock_guard<std::mutex> lk(_data_mu);
        if (_data_err.empty()) return;
        _data_answer = retry ? 1 : 2;
    }
    _data_cv.notify_all();
}

}  // namespace gui
