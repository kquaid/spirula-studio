#pragma once

#include "config/TrainConfig.h"
#include <string>

namespace spirula {

class GuiApp;

class MultiframeTrainPanel {
public:
    MultiframeTrainPanel() = default;

    // Called once during GUI initialization
    void init(GuiApp* app);

    // Rendered each frame when the Multiframe Training tab is selected
    void render();

    // Accessors for SOG settings
    bool export_sog_enabled() const { return _export_sog; }
    void set_export_sog(bool enabled) { _export_sog = enabled; }

private:
    GuiApp* _app = nullptr;
    TrainConfig _config;

    // SOG export toggles
    bool _export_sog = true;
    int  _sog_texture_resolution = 1024; // Default to 1024x1024 grid
    std::string _output_sog_path;

    // Helper to render the shared training options
    void draw_training_options();
    void draw_sog_export_section();
};

} // namespace spirula