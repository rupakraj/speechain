${SPEECHAIN_PYTHON} ${SPEECHAIN_ROOT}/datasets/pyscripts/vocab_generator.py \
    --text_path ${SPEECHAIN_ROOT}/datasets/slr54nepaliasr/data/textonly \
    --save_path ${SPEECHAIN_ROOT}/datasets/slr54nepaliasr/data/sentencepiece/train \
    --token_type sentencepiece \
    --txt_format no-punc \
    --model_type bpe \
    --character_coverage 1.0 \
    --vocab_size 1000 \
    --split_by_whitespace true\

# --vocab_size ${vocab_size}"
# --model_type ${model_type} => bpe or unigram
# --character_coverage ${character_coverage} --split_by_whitespace ${split_by_whitespace}"
