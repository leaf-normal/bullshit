#include "Film.h"

Film::Film(grassland::graphics::Core* core, int width, int height)
    : core_(core)
    , width_(width)
    , height_(height)
    , sample_count_(0) {
    
    CreateImages();
    Reset();
}

Film::~Film() {
    accumulated_color_image_.reset();
    accumulated_samples_image_.reset();
    output_image_.reset();
}

void Film::CreateImages() {
    core_->CreateImage(width_, height_, 
                      grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                      &accumulated_color_image_);
    
    core_->CreateImage(width_, height_, 
                      grassland::graphics::IMAGE_FORMAT_R32_SINT,
                      &accumulated_samples_image_);
    
    core_->CreateImage(width_, height_, 
                      grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                      &output_image_);
}

void Film::Reset() {
    std::unique_ptr<grassland::graphics::CommandContext> cmd_context;
    core_->CreateCommandContext(&cmd_context);
    cmd_context->CmdClearImage(accumulated_color_image_.get(), { {0.0f, 0.0f, 0.0f, 0.0f} });
    cmd_context->CmdClearImage(accumulated_samples_image_.get(), { {0, 0, 0, 0} });
    cmd_context->CmdClearImage(output_image_.get(), { {0.0f, 0.0f, 0.0f, 0.0f} });
    core_->SubmitCommandContext(cmd_context.get());
    
    sample_count_ = 0;
    grassland::LogInfo("Film accumulation reset");
}

void Film::DevelopToOutput() {
    
    if (sample_count_ == 0) {
        return;
    }

    // Download accumulated color and samples
    size_t color_size = width_ * height_ * sizeof(float) * 4;
    std::vector<float> accumulated_colors(width_ * height_ * 4);
    accumulated_color_image_->DownloadData(accumulated_colors.data());

    // Divide by sample count to get average
    std::vector<float> output_colors(width_ * height_ * 4);
    for (int i = 0; i < width_ * height_ * 4; i++) {
        output_colors[i] = accumulated_colors[i] / static_cast<float>(sample_count_);
    }

    // Upload to output image
    output_image_->UploadData(output_colors.data());
}

void Film::Resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }

    width_ = width;
    height_ = height;

    // Recreate images with new dimensions
    accumulated_color_image_.reset();
    accumulated_samples_image_.reset();
    output_image_.reset();

    CreateImages();
    Reset();
    
    grassland::LogInfo("Film resized to {}x{}", width, height);
}

