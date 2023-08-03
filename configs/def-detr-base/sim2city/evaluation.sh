BATCH_SIZE=16
DATA_ROOT=<YOUR/DATA/ROOT>
OUTPUT_DIR=./outputs/def-detr-base/sim2city/evaluation

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../teaching/model_best.pth

