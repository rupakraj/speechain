# Small model ne_char_conformer-small_lr2e-3
# Small V2 model ne_char_conformer-small-v2_lr2e-3
# Medium model ne_char_conformer-medium_lr2e-3
# FineTune model ne_char_conformer-tuned_lr2e-3
# FineTune model ne_char_conformer-large-v2_lr2e-3
# ne_char_conformer-large-v2_lr2e-3-ccnn-lm
# ne_char_conformer-small-v2_lr2e-3-ccnn-lm
# ne_char_conformer-large-v2_lr2e-3-mix

bash run.sh \
    --exp_cfg ne_char_conformer-small-v2_lr2e-3 \
    --train_num_workers 6 \
    --valid_num_workers 6 \
    --test_num_workers 6 \
    --ngpu 2 \
    --train false \
    --test true \
    --resume false \
    --test_model latest \
