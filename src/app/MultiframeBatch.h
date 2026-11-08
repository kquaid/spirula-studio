#pragma once

// Multiframe batch: frame N's trained splats seed frame N+1, which refines
// them for a few iterations with the splat count and order frozen. Frames
// live in sibling folders <root>/<n> and share the first frame's cameras.
// Each frame ends as <output>/ply/frame_<n>.ply and <output>/sog/frame_<n>.sog.

#include "app/TrainerCore.h"

#include <functional>
#include <string>
#include <vector>

namespace spirula {

struct MultiframeSettings {
    int first_frame = -1;            // -1: the opened folder's number
    int last_frame = -1;             // -1: every consecutive folder present
    int frame_iterations = 100;      // for every frame after the first
    float lr_scale = 1.0f;           // on the normal per-parameter rates, held constant
    std::string output_dir;          // empty: <root>/multiframe_out
};

struct MultiframeHooks {
    // A frame's engine is set up; engine_mutex is held. Viewer uploads go here.
    std::function<void(int frame, int index, int total)> engine_ready;
    // A frame's PLY is on disk and its SOG is queued.
    std::function<void(int frame, int index, int total)> frame_done;
};

// The folder name as a frame number, or -1 when it is not one.
int multiframe_frame_number(const std::string& folder_name);

// The frames to run, in order. Throws when a folder is missing: any in an
// explicit range, or the first of an open-ended one.
std::vector<int> multiframe_frames(const std::string& root, int first, int last);

// The config every frame after the first trains with.
TrainConfig multiframe_refine_config(const TrainConfig& base, const MultiframeSettings& m);

// Runs the batch on `s`, which has parsed the first frame's folder but not
// set up its engine. Returns early, without that frame's outputs, when
// s.stop_requested is set. Throws on a missing frame or file.
void run_multiframe(TrainerSession& s, const MultiframeSettings& m,
                    const MultiframeHooks& hooks, const TrainerCallbacks& cb);

}  // namespace spirula
