#include "app/gui/MultiframeTrainPanel.h"
#include "app/gui/Ui.h"
#include "config/TrainConfig.h"
#include "imgui.h"

namespace spirula {

void MultiframeTrainPanel::init(GuiApp* app) {
    _app = app;
    // Initialize defaults from the base TrainConfig
    _config = TrainConfig();
}

void MultiframeTrainPanel::draw_sog_export_section() {
    if (ImGui::CollapsingHeader("SOG Export Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Export SOG (.sog) on Completion", &_export_sog);
        
        if (_export_sog) {
            ImGui::Indent();
            
            const char* resolutions[] = { "512 x 512 (262k)", "1024 x 1024 (1M)", "2048 x 2048 (4.1M)" };
            static int res_idx = 1; // 1024
            if (ImGui::Combo("SOG Grid Size", &res_idx, resolutions, IM_ARRAYSIZE(resolutions))) {
                if (res_idx == 0) _sog_texture_resolution = 512;
                if (res_idx == 1) _sog_texture_resolution = 1024;
                if (res_idx == 2) _sog_texture_resolution = 2048;
            }

            ui::TextRaw("SOG format arranges splats along a 3D Morton space-filling curve.");
            ui::TextRaw("Suitable for web viewing in PlayCanvas / SuperSplat.");
            ImGui::Unindent();
        }
    }
}

void MultiframeTrainPanel::draw_training_options() {
    if (ImGui::CollapsingHeader("Training Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputInt("Iterations", &_config.num_iterations);
        ImGui::SliderInt("SH Degree", &_config.sh_degree, 0, 3);
        ImGui::Checkbox("Enable Bilateral Grid (Color correction)", &_config.use_bilateral_grid);
    }
}

void MultiframeTrainPanel::render() {
    ui::TextRaw("Multiframe Gaussian Splatting Training");
    ImGui::Separator();
    ImGui::Spacing();

    // 1. Draw base training options
    draw_training_options();
    ImGui::Spacing();

    // 2. Draw SOG export options
    draw_sog_export_section();
    ImGui::Spacing();

    // 3. Action buttons
    if (ui::ButtonRaw("Start Multiframe Training")) {
        // Trigger training run (we will wire the runner in the next commit)
    }
}

} // namespace spirula