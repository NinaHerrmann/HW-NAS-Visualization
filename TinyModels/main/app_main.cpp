#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_cls_postprocessor.hpp"
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
#include "esp_sleep.h"
#include "dl_image_jpeg.hpp"
#include <cmath>

extern const uint8_t model_espdl[] asm("_binary_model3043_espdl_start");


extern const uint8_t picture_start[] asm("_binary_cifar10_jpg_start");
extern const uint8_t picture_end[] asm("_binary_cifar10_jpg_end");


extern "C" void app_main(void)
{
    dl::image::jpeg_img_t jpeg_img = {
        .data = (uint8_t *)picture_start,
        .data_len = (uint32_t)(picture_end - picture_start),
    };
    const dl::image::img_t img = sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    // char dir[64];
    // snprintf(dir, sizeof(dir), "%s/espdl_models", CONFIG_BSP_SD_MOUNT_POINT);
    dl::Model* model = new dl::Model((const char *)model_espdl);

    const std::array<float, 3> mean_vals{0.4914f, 0.4822f, 0.4465f};
    const std::array<float, 3> std_vals{0.2470f, 0.2435f, 0.2616f};

    dl::image::ImagePreprocessor* m_image_preprocessor = new dl::image::ImagePreprocessor(model, mean_vals, std_vals);
    m_image_preprocessor->preprocess(img);
    model->run(dl::RUNTIME_MODE_MULTI_CORE);

    // ESP_LOGI("NINA:","Finished\n.");
    delete model;  
}
