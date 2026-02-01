# OSLR 54 dataset for Nepali Speech Recognition

__TODO__

## Build dataset

```bash
# verify the path
echo $SPEECHAIN_ROOT
echo $SPEECHAIN_PYTHON

cd $SPEECHAIN_ROOT/datasets/slr54nepaliasr
bash run.sh --src_path $SPEECHAIN_ROOT/datasets/slr54nepaliasr --ncpu 24 --ngpu 2
```