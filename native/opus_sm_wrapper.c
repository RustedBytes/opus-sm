#include <stdlib.h>

#include "analysis.h"
#include "opus_custom.h"
#include "opus_private.h"

typedef struct OpusSmAnalyzer {
    TonalityAnalysisState analysis;
    CELTMode *mode;
} OpusSmAnalyzer;

OpusSmAnalyzer *opus_sm_analyzer_create(opus_int32 sample_rate) {
    OpusSmAnalyzer *analyzer = (OpusSmAnalyzer *)calloc(1, sizeof(OpusSmAnalyzer));
    if (analyzer == NULL) {
        return NULL;
    }

    analyzer->mode = opus_custom_mode_create(48000, 960, NULL);
    if (analyzer->mode == NULL) {
        free(analyzer);
        return NULL;
    }

    tonality_analysis_init(&analyzer->analysis, sample_rate);
    return analyzer;
}

void opus_sm_analyzer_destroy(OpusSmAnalyzer *analyzer) {
    if (analyzer == NULL) {
        return;
    }
    free(analyzer);
}

void opus_sm_analyzer_reset(OpusSmAnalyzer *analyzer) {
    if (analyzer == NULL) {
        return;
    }

    tonality_analysis_reset(&analyzer->analysis);
}

int opus_sm_analyzer_process(
    OpusSmAnalyzer *analyzer,
    const float *pcm,
    int frame_size,
    int channels,
    float *music_prob,
    float *activity_prob
) {
    AnalysisInfo info = {0};

    if (analyzer == NULL || analyzer->mode == NULL || pcm == NULL || frame_size <= 0 || channels <= 0) {
        return 0;
    }
    run_analysis(
        &analyzer->analysis,
        analyzer->mode,
        pcm,
        frame_size,
        frame_size,
        0,
        -2,
        channels,
        48000,
        24,
        downmix_float,
        &info
    );

    if (music_prob != NULL) {
        *music_prob = info.music_prob;
    }
    if (activity_prob != NULL) {
        *activity_prob = info.activity_probability;
    }

    return info.valid;
}
